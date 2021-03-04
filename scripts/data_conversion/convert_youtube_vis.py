import sys
import os
from glob import glob
import json
from pycocotools.mask import frPyObjects, decode
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class YouTubeVISConverter(_BaseDatasetConverter):
    """Converter for YouTubeVIS ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/youtube_vis/'),
            # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/youtube_vis/'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'train_sub_split',  # Split to convert
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
        self.gt_fol = config['ORIGINAL_GT_FOLDER'] + 'youtube_vis_' + self.config['SPLIT_TO_CONVERT']
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']

        # load gt data
        gt_dir_files = glob(os.path.join(self.gt_fol, '*.json'))
        assert len(gt_dir_files) == 1, self.gt_fol + ' does not contain exactly one json file.'

        with open(gt_dir_files[0]) as f:
            self.gt_data = json.load(f)

        self.class_name_to_class_id = {cat['name']: cat['id'] for cat in self.gt_data['categories']}

        # determine sequences
        self.sequences = {vid['file_names'][0].split('/')[0]: vid['id'] for vid in self.gt_data['videos']}
        self.seq_list = list(self.sequences.keys())
        self.seq_lengths = {vid['file_names'][0].split('/')[0]: vid['length'] for vid in self.gt_data['videos']}
        self.seq_sizes = {vid['file_names'][0].split('/')[0]: (vid['height'], vid['width'])
                          for vid in self.gt_data['videos']}

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in self.seq_list:
            # determine annotations for given sequence
            seq_id = self.sequences[seq]
            seq_annotations = [ann for ann in self.gt_data['annotations'] if ann['video_id'] == seq_id]
            lines = []
            for t in range(self.seq_lengths[seq]):
                for ann in seq_annotations:
                    if ann['segmentations'][t]:
                        h = ann['height']
                        w = ann['width']
                        mask_encoded = frPyObjects(ann['segmentations'][t], h, w)
                        lines.append('%d %d %d %d %d %s %d %d %d %d %d %d %d %d\n'
                                     % (t, ann['id'], ann['category_id'], h, w, mask_encoded['counts'].decode("utf-8"),
                                        0, 0, 0, 0, ann['iscrowd'], 0, 0, 0))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = YouTubeVISConverter.get_default_config()
    conf = utils.update_config(default_conf)
    YouTubeVISConverter(conf).convert()
