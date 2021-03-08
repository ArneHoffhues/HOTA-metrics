import sys
import os
import csv
from _base_tracker_data_converter import _BaseTrackerDataConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class MOTSChallengeTrackerConverter(_BaseTrackerDataConverter):
    """Converter for MOTChallengeMOTS tracker data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_TRACKER_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),
            # Location of original GT data
            'NEW_TRACKER_FOLDER': os.path.join(code_path, 'data/converted_trackers/mot_challenge/'),
            # Location for the converted GT data
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt/mot_challenge/'),
            # Location of ground truth data where the seqmap and the clsmap reside
            'SPLIT_TO_CONVERT': 'train',  # Split to convert
            'TRACKER_LIST': None,  # List of trackers to convert, None for all in folder
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'MOTSChallenge'

    def __init__(self, config):
        super().__init__()
        self.config = config
        benchmark = 'MOTS-' + config['SPLIT_TO_CONVERT']
        self.tracker_fol = os.path.join(config['ORIGINAL_TRACKER_FOLDER'], benchmark)
        self.new_tracker_folder = os.path.join(config['NEW_TRACKER_FOLDER'], benchmark)
        if not config['TRACKER_LIST']:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            tracker_dirs = os.listdir(self.tracker_fol)
            for tracker in self.tracker_list:
                assert tracker in tracker_dirs, 'Tracker directory for tracker %s missing in %s' \
                                                % (tracker, self.tracker_fol)
            self.tracker_list = config['TRACKER_LIST']
        self.gt_dir = config['GT_FOLDER']
        self.split_to_convert = benchmark
        self.output_as_zip = config['OUTPUT_AS_ZIP']

        # Get sequences
        self._get_sequences()

        # check if all tracker files are present
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.txt')
                if not os.path.isfile(curr_file):
                    raise Exception('Tracker file %s not found for tracker %s.'
                                    % (curr_file, tracker))

        # Get classes
        self._get_classes()

    def _prepare_data(self, tracker):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in self.seq_list:
            # load sequence
            file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.txt')

            lines = []
            with open(file) as fp:
                dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')  # Auto determine file structure.
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    lines.append('%d %s %s %s %s %s %f %f %f %f %f\n'
                                 % (int(row[0]) - 1, row[1], row[2], row[3], row[4], row[5], 0, 0, 0, 0, 1))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = MOTSChallengeTrackerConverter.get_default_config()
    conf = utils.update_config(default_conf)
    MOTSChallengeTrackerConverter(conf).convert()
