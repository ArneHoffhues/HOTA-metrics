import sys
import os
from glob import glob
import json
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class BDD100KConverter(_BaseDatasetConverter):
    """Converter for BDD100K ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/bdd100k/'),  # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/bdd100k/'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'val',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'BDD100K'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = config['ORIGINAL_GT_FOLDER'] + config['SPLIT_TO_CONVERT']
        self.new_gt_folder = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
        self.class_name_to_class_id = {'pedestrian': 1, 'rider': 2, 'other person': 3, 'car': 4, 'bus': 5, 'truck': 6,
                                       'train': 7, 'trailer': 8, 'other vehicle': 9, 'motorcycle': 10, 'bicycle': 11}

        # Get sequences
        self.sequences = glob(os.path.join(self.gt_fol, '*.json'))
        self.seq_list = [seq.split('/')[-1].split('.')[0] for seq in self.sequences]
        self.seq_lengths = {}

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in self.seq_list:
            # load sequence
            seq_file = os.path.join(self.gt_fol, seq + '.json')
            with open(seq_file) as f:
                gt_data = json.load(f)
            # order by timestep
            gt_data = sorted(gt_data, key=lambda x: x['index'])
            num_timesteps = len(gt_data)
            self.seq_lengths[seq] = num_timesteps

            lines = []
            for t in range(num_timesteps):
                for label in gt_data[t]['labels']:
                    if 'attributes' in label.keys():
                        is_crowd = int(label['attributes']['Crowd'])
                        is_truncated = int(label['attributes']['Truncated'])
                        is_occluded = int(label['attributes']['Occluded'])
                    else:
                        is_crowd = 0
                        is_truncated = 0
                        is_occluded = 0
                    lines.append('%d %d %d %d %d %d %d %d %d %s %f %f %f %f\n'
                                 % (t, int(label['id']), self.class_name_to_class_id[label['category']], is_crowd,
                                    is_truncated, is_occluded, 0, 0, 0, 'None', label['box2d']['x1'],
                                    label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = BDD100KConverter.get_default_config()
    conf = utils.update_config(default_conf)
    BDD100KConverter(conf).convert()
