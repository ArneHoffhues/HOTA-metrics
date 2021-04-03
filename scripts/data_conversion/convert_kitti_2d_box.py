import sys
import os
import csv
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from trackeval import utils  # noqa: E402


class Kitti2DBoxConverter(_BaseDatasetConverter):
    """Converter for Kitti 2D Box ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box_train'),
            # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/kitti/'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'training',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'Kitti2DBox'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = config['ORIGINAL_GT_FOLDER']
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = 'kitti_2d_box_' + config['SPLIT_TO_CONVERT']
        # class list and corresponding class ids
        self.class_name_to_class_id = {'car': 1, 'van': 2, 'truck': 3, 'pedestrian': 4, 'person': 5,
                                       'cyclist': 6, 'tram': 7, 'misc': 8, 'dontcare': 9}

        # Get sequences to convert and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        self.seq_sizes = {}
        seqmap_name = 'evaluate_tracking.seqmap.' + config['SPLIT_TO_CONVERT']
        seqmap_file = os.path.join(self.gt_fol, seqmap_name)
        if not os.path.isfile(seqmap_file):
            raise Exception('no seqmap found: ' + os.path.basename(seqmap_file))
        with open(seqmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.read(1024))
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for row in reader:
                if len(row) >= 4:
                    seq = row[0]
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[3])
                    # no sequence size informaton for Kitti
                    self.seq_sizes[seq] = (0, 0)
                    curr_file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')
                    if not os.path.isfile(curr_file):
                        raise Exception('GT file not found: ' + os.path.basename(curr_file))

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in self.seq_list:
            # sequence path
            file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')

            lines = []
            with open(file) as fp:
                dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=' ')  # Auto determine file structure.
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    lines.append('%s %s %d %d %d %s %s %s %s %s %d %s %s %d\n'
                                 % (row[0], row[1], self.class_name_to_class_id[row[2].lower()], 0, 0, 'None', row[6],
                                    row[7], row[8], row[9], 0, row[3], row[4], 0))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = Kitti2DBoxConverter.get_default_config()
    conf = utils.update_config(default_conf)
    Kitti2DBoxConverter(conf).convert()
