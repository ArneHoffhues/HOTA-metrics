import sys
import os
import json
import itertools
import numpy as np
from _base_tracker_data_converter import _BaseTrackerDataConverter
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class TAOTrackerConverter(_BaseTrackerDataConverter):
    """Converter for TAO tracker data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_TRACKER_FOLDER': os.path.join(code_path, 'data/trackers/tao/'),
            # Location of original GT data
            'NEW_TRACKER_FOLDER': os.path.join(code_path, 'data/converted_trackers/tao'),
            # Location for the converted GT data
            'GT_DATA_LOC': os.path.join(code_path, 'data/gt/tao/'),  # Folder of the original gt data
            'SPLIT_TO_CONVERT': 'training',  # Split to convert
            'TRACKER_LIST': None,  # List of trackers to convert, None for all in folder
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'TAO'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tracker_fol = os.path.join(config['ORIGINAL_TRACKER_FOLDER'], 'tao_' + config['SPLIT_TO_CONVERT'])
        self.new_tracker_folder = os.path.join(config['NEW_TRACKER_FOLDER'], config['SPLIT_TO_CONVERT'])
        self.orig_gt_folder = os.path.join(config['GT_DATA_LOC'], 'tao_' + config['SPLIT_TO_CONVERT'])
        if not config['TRACKER_LIST']:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            tracker_dirs = os.listdir(self.tracker_fol)
            for tracker in self.tracker_list:
                assert tracker in tracker_dirs, 'Tracker directory for tracker %s missing in %s' \
                                                % (tracker, self.tracker_fol)
            self.tracker_list = config['TRACKER_LIST']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
        self.output_as_zip = config['OUTPUT_AS_ZIP']

        # read gt file
        gt_dir_files = glob(os.path.join(self.orig_gt_folder, '*.json'))
        assert len(gt_dir_files) == 1, self.orig_gt_folder + ' does not contain exactly one json file.'

        with open(gt_dir_files[0]) as f:
            self.gt_data = json.load(f)

        # get sequences
        self.seq_to_seqid = {vid['name'].replace('/', '-'): vid['id'] for vid in self.gt_data['videos']}
        self.seq_list = list(self.seq_to_seqid.keys())
        self.seq_sizes = {vid['id']: (vid['height'], vid['width']) for vid in self.gt_data['videos']}
        self.seq_lengths = {}
        vids_to_images = {vid['id']: [] for vid in self.gt_data['videos']}
        for img in self.gt_data['images']:
            vids_to_images[img['video_id']].append(img)
        self.vids_to_images = {k: sorted(v, key=lambda x: x['frame_index']) for k, v in vids_to_images.items()}
        for vid, imgs in vids_to_images.items():
            self.seq_lengths[vid] = len(imgs)

        # check if all tracker files are present
        for tracker in self.tracker_list:
            tracker_dir_files = glob(os.path.join(self.tracker_fol, tracker, 'data', '*.json'))
            assert len(tracker_dir_files) == 1, os.path.join(self.tracker_fol,tracker) + \
                                                ' does not contain exactly one json file.'


    def _prepare_data(self, tracker):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}

        # load tracker data
        tracker_dir_files = glob(os.path.join(self.tracker_fol, tracker,'data', '*.json'))

        with open(tracker_dir_files[0]) as f:
            tracker_data = json.load(f)

        # ensure that video ids are present for all tracker data
        self._fill_video_ids_inplace(tracker_data)
        # make track ids unique
        self._make_track_ids_unique(tracker_data)

        for seq in tqdm(self.seq_list):
            seq_id = self.seq_to_seqid[seq]
            # determine annotations for given sequence
            seq_annotations = [ann for ann in tracker_data if ann['video_id'] == seq_id]

            lines = []
            for t in range(self.seq_lengths[seq_id]):
                # determine annotations for given timestep
                timestep_annotations = [ann for ann in seq_annotations
                                        if ann['image_id'] == self.vids_to_images[seq_id][t]['id']]
                for ann in timestep_annotations:
                    # compute smallest mask which fully covers the bounding box
                    # mask = np.zeros(self.seq_sizes[seq_id], order='F').astype(np.uint8)
                    # mask[int(np.floor(ann['bbox'][1])):int(np.ceil(ann['bbox'][1] + ann['bbox'][3])+1),
                    # int(np.floor(ann['bbox'][0])):int(np.ceil(ann['bbox'][0] + ann['bbox'][2])+1)] = 1
                    # encoded_mask = mask_utils.encode(mask)
                    # # convert box format from xywh to x0y0x1y1
                    # lines.append('%d %d %d %d %d %s %.13f %.13f %.13f %.13f %.20f\n'
                    #              % (t, ann['track_id'], ann['category_id'], encoded_mask['size'][0],
                    #                 encoded_mask['size'][1], encoded_mask['counts'], ann['bbox'][0], ann['bbox'][1],
                    #                 ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3], ann['score']))
                    lines.append('%d %d %d %d %d %s %.13f %.13f %.13f %.13f %.20f\n'
                                 % (t, ann['track_id'], ann['category_id'], 0,
                                    0, 'None', ann['bbox'][0], ann['bbox'][1],
                                    ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3], ann['score']))
            data[seq] = lines
        return data

    def _fill_video_ids_inplace(self, annotations):
        """
        taken from https: // github.com / TAO - Dataset / tao / blob / master / tao / utils / evaluation.py
        fills missing video ids in tracker data inplace
        """
        missing_video_id = [x for x in annotations if 'video_id' not in x]
        if missing_video_id:
            image_id_to_video_id = {
                x['id']: x['video_id'] for x in self.gt_data['images']
            }
            for x in missing_video_id:
                x['video_id'] = image_id_to_video_id[x['image_id']]

    @staticmethod
    def _make_track_ids_unique(annotations):
        """
        taken from https://github.com/TAO-Dataset/tao/blob/master/tao/utils/evaluation.py
        makes track ids unique over the whole evaluation set and returns the number of updated track ids
        """
        track_id_videos = {}
        track_ids_to_update = set()
        max_track_id = 0
        for ann in annotations:
            t = ann['track_id']
            if t not in track_id_videos:
                track_id_videos[t] = ann['video_id']

            if ann['video_id'] != track_id_videos[t]:
                # Track id is assigned to multiple videos
                track_ids_to_update.add(t)
            max_track_id = max(max_track_id, t)

        if track_ids_to_update:
            print('true')
            next_id = itertools.count(max_track_id + 1)
            new_track_ids = defaultdict(lambda: next(next_id))
            for ann in annotations:
                t = ann['track_id']
                v = ann['video_id']
                if t in track_ids_to_update:
                    ann['track_id'] = new_track_ids[t, v]
        return len(track_ids_to_update)

if __name__ == '__main__':
    default_conf = TAOTrackerConverter.get_default_config()
    conf = utils.update_config(default_conf)
    TAOTrackerConverter(conf).convert()