import sys
import os
import json
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class OVISConverter(_BaseDatasetConverter):
    """Converter for OVIS ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/ovis/'),
            # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/ovis/'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'train',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'OVIS'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = config['ORIGINAL_GT_FOLDER'] + 'ovis_' + self.config['SPLIT_TO_CONVERT']
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = 'ovis_' + config['SPLIT_TO_CONVERT']

        # load gt data
        gt_dir_files = [file for file in os.listdir(self.gt_fol) if file.endswith('.json')]
        if len(gt_dir_files) != 1:
            raise Exception(self.gt_fol + ' does not contain exactly one json file.')

        with open(os.path.join(self.gt_fol, gt_dir_files[0])) as f:
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
                        h = ann['segmentations'][t]['size'][0]
                        w = ann['segmentations'][t]['size'][1]
                        mask = ann['segmentations'][t]['counts']
                        lines.append('%d %d %d %d %d %s %f %f %f %f %d %d %d %d\n'
                                     % (t, ann['id'], ann['category_id'], h, w, mask,
                                        0, 0, 0, 0, ann['iscrowd'], 0, 0, 0))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = OVISConverter.get_default_config()
    conf = utils.update_config(default_conf)
    OVISConverter(conf).convert()