import sys
import os
import csv
import configparser
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class MOTSChallengeConverter(_BaseDatasetConverter):
    """Converter for MOTChallenge MOTS ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/mot_challenge/mots_challenge'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'train',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'MOTChallengeMOTS'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_set = 'MOTS-' + config['SPLIT_TO_CONVERT']
        self.gt_fol = config['ORIGINAL_GT_FOLDER'] + self.gt_set
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
        self.class_name_to_class_id = {'pedestrian': 2, 'ignore': 10}

        # Get sequences to convert and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        self.seq_sizes = {}
        seqmap_file = os.path.join(config['ORIGINAL_GT_FOLDER'], 'seqmaps', self.gt_set + '.txt')
        assert os.path.isfile(seqmap_file), 'no seqmap found: ' + seqmap_file
        with open(seqmap_file) as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0 or row[0] == '':
                    continue
                seq = row[0]
                self.seq_list.append(seq)
                ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                assert os.path.isfile(ini_file), 'ini file does not exist: ' + ini_file
                ini_data = configparser.ConfigParser()
                ini_data.read(ini_file)
                self.seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                self.seq_sizes[seq] = (int(ini_data['Sequence']['imHeight']), int(ini_data['Sequence']['imWidth']))
                curr_file = os.path.join(self.gt_fol, seq, 'gt', 'gt.txt')
                assert os.path.isfile(curr_file), 'GT file not found: ' + curr_file

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in self.seq_list:
            # sequence path
            file = os.path.join(self.gt_fol, seq, 'gt', 'gt.txt')

            lines = []
            with open(file) as fp:
                dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=' ')  # Auto determine file structure.
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    lines.append('%s %s %s %s %s %s %f %f %f %f %d %d %d %d\n'
                                 % (row[0], row[1], row[2], row[3], row[4], row[5], 0, 0, 0, 0, 0, 0, 0, 0))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = MOTSChallengeConverter.get_default_config()
    conf = utils.update_config(default_conf)
    MOTSChallengeConverter(conf).convert()
