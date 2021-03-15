import sys
import os
import json
from _base_tracker_data_converter import _BaseTrackerDataConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class YouTubeVISTrackerConverter(_BaseTrackerDataConverter):
    """Converter for YouTubeVIS tracker data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_TRACKER_FOLDER': os.path.join(code_path, 'data/trackers/youtube_vis/'),
            # Location of original GT data
            'NEW_TRACKER_FOLDER': os.path.join(code_path, 'data/converted_trackers/youtube_vis/'),
            # Location for the converted GT data
            'GT_DATA_LOC': os.path.join(code_path, 'data/gt/youtube_vis/'),
            # Folder of the original gt data
            'SPLIT_TO_CONVERT': 'train_sub_split',  # Split to convert
            'TRACKER_LIST': None,  # List of trackers to convert, None for all in folder
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'YouTubeVIS'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split_to_convert = 'youtube_vis_' + config['SPLIT_TO_CONVERT']
        self.tracker_fol = os.path.join(config['ORIGINAL_TRACKER_FOLDER'], self.split_to_convert)
        self.new_tracker_folder = os.path.join(config['NEW_TRACKER_FOLDER'], self.split_to_convert)
        self.orig_gt_folder = os.path.join(config['GT_DATA_LOC'], self.split_to_convert)
        if not config['TRACKER_LIST']:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            tracker_dirs = os.listdir(self.tracker_fol)
            for tracker in self.tracker_list:
                assert tracker in tracker_dirs, 'Tracker directory for tracker %s missing in %s' \
                                                % (tracker, self.tracker_fol)
            self.tracker_list = config['TRACKER_LIST']
        self.output_as_zip = config['OUTPUT_AS_ZIP']

        self.id_counter = 0

        # read gt file
        gt_dir_files = [file for file in os.listdir(self.orig_gt_folder) if file.endswith('.json')]
        if len(gt_dir_files) != 1:
            raise Exception(self.orig_gt_folder + ' does not contain exactly one json file.')

        with open(os.path.join(self.orig_gt_folder, gt_dir_files[0])) as f:
            self.gt_data = json.load(f)

        # get sequences
        self.seq_to_seqid = {vid['file_names'][0].split('/')[0]: vid['id'] for vid in self.gt_data['videos']}
        self.seq_list = list(self.seq_to_seqid.keys())
        self.seq_lengths = {vid['file_names'][0].split('/')[0]: vid['length'] for vid in self.gt_data['videos']}

        # check if all tracker files are present
        for tracker in self.tracker_list:
            tracker_dir_path = os.path.join(self.tracker_fol, tracker, 'data')
            tr_dir_files = [file for file in os.listdir(tracker_dir_path) if file.endswith('.json')]
            if len(tr_dir_files) != 1:
                raise Exception(tracker_dir_path + ' does not contain exactly one json file.')


    def _prepare_data(self, tracker):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}

        # load tracker data
        tracker_dir_path = os.path.join(self.tracker_fol, tracker, 'data')
        tr_dir_files = [file for file in os.listdir(tracker_dir_path) if file.endswith('.json')]

        with open(os.path.join(tracker_dir_path, tr_dir_files[0])) as f:
            tracker_data = json.load(f)

        # add track id to each track
        for track in tracker_data:
            track['id'] = self.id_counter
            self.id_counter += 1

        for seq in self.seq_list:
            # determine annotations for given sequence
            seq_id = self.seq_to_seqid[seq]

            seq_annotations = [ann for ann in tracker_data if ann['video_id'] == seq_id]
            lines = []
            for t in range(self.seq_lengths[seq]):
                for ann in seq_annotations:
                    if ann['segmentations'][t]:
                        mask = ann['segmentations'][t]
                        lines.append('%d %d %d %d %d %s %f %f %f %f %.20f\n'
                                     % (t, ann['id'], ann['category_id'], mask['size'][0], mask['size'][1],
                                        mask['counts'], 0, 0, 0, 0, ann['score']))
            data[seq] = lines
        return data

if __name__ == '__main__':
    default_conf = YouTubeVISTrackerConverter.get_default_config()
    conf = utils.update_config(default_conf)
    YouTubeVISTrackerConverter(conf).convert()
