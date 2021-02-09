import sys
import os
import csv
import configparser
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class MOTChallenge2DBoxConverter(_BaseDatasetConverter):
    """Converter for MOTChallenge 2D Box ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/mot_challenge/mot_challenge_2d_box'),
            # Location for the converted GT data
            'BENCHMARK': 'MOT17',  # Benchmark to convert
            'SPLIT_TO_CONVERT': 'train',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'MOTChallenge2DBox'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_set = config['BENCHMARK'] + '-' + config['SPLIT_TO_CONVERT']
        self.gt_fol = config['ORIGINAL_GT_FOLDER'] + self.gt_set
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
        self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                       'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                       'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}

        # Get sequences to convert and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        self.seq_sizes = {}
        seqmap_file = os.path.join(config['ORIGINAL_GT_FOLDER'], 'seqmaps', self.gt_set + '.txt')
        if not os.path.isfile(seqmap_file):
            raise Exception('no seqmap found: ' + os.path.basename(seqmap_file))
        with open(seqmap_file) as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0 or row[0] == '':
                    continue
                seq = row[0]
                self.seq_list.append(seq)
                ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                if not os.path.isfile(ini_file):
                    raise Exception('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                ini_data = configparser.ConfigParser()
                ini_data.read(ini_file)
                self.seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                self.seq_sizes[seq] = (int(ini_data['Sequence']['imHeight']), int(ini_data['Sequence']['imWidth']))
                curr_file = os.path.join(self.gt_fol, seq, 'gt', 'gt.txt')
                if not os.path.isfile(curr_file):
                    raise Exception('GT file not found: ' + seq + '/gt/' + os.path.basename(curr_file))

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
                dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=',')  # Auto determine file structure.
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                # convert box format from xywh to x0y0x1y1
                for row in reader:
                    lines.append('%s %s %s %d %d %d %s %d %d %s %f %f %f %f\n'
                                 % (row[0], row[1], row[7], 0, 0, 0, row[6], 0, 0, 'None', float(row[2]), float(row[3]),
                                    float(row[2]) + float(row[4]), float(row[3]) + float(row[5])))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = MOTChallenge2DBoxConverter.get_default_config()
    conf = utils.update_config(default_conf)
    MOTChallenge2DBoxConverter(conf).convert()
