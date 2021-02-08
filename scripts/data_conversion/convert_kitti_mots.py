import sys
import os
import csv
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class KittiMOTSConverter(_BaseDatasetConverter):
    """Converter for Kitti MOTS ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_mots'),  # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/kitti/kitti_mots'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'val',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'KittiMOTS'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = config['ORIGINAL_GT_FOLDER']
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
        self.class_name_to_class_id = {'cars': 1, 'pedestrians': 2, 'ignore': 10}

        # Get sequences to convert and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        seqmap_name = config['SPLIT_TO_CONVERT'] + ".seqmap"
        seqmap_file = os.path.join(self.gt_fol, seqmap_name)
        assert os.path.isfile(seqmap_file), 'no seqmap %s found in %s' % (seqmap_name, self.gt_fol)

        with open(seqmap_file, "r") as fp:
            dialect = csv.Sniffer().sniff(fp.read(1024))
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for row in reader:
                if len(row) >= 4:
                    seq = "%04d" % int(row[0])
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[3]) + 1
                    assert os.path.isfile(os.path.join(self.gt_fol, 'instances_txt', seq + '.txt')), \
                        'GT file %s.txt not found in %s' % (seq, os.path.join(self.gt_fol, 'instances_txt'))

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in self.seq_list:
            # load sequences
            file = os.path.join(self.gt_fol, 'instances_txt', seq + '.txt')

            lines = []
            with open(file) as fp:
                dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=' ')  # Auto determine file structure.
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    lines.append('%s %s %s %d %d %d %d %s %s %s %f %f %f %f\n'
                                 % (row[0], row[1], row[2], 0, 0, 0, 0, row[3], row[4], row[5], 0, 0, 0, 0))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = KittiMOTSConverter.get_default_config()
    conf = utils.update_config(default_conf)
    KittiMOTSConverter(conf).convert()
