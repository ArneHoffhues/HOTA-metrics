import sys
import os
import csv
from _base_tracker_data_converter import _BaseTrackerDataConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class Kitti2DBoxTrackerConverter(_BaseTrackerDataConverter):
    """Converter for Kitti 2D Box tracker data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_TRACKER_FOLDER': os.path.join(code_path, 'data/trackers/kitti/kitti_2d_box_train'),
            # Location of original GT data
            'NEW_TRACKER_FOLDER': os.path.join(code_path, 'data/converted_trackers/kitti/kitti_2d_box'),
            # Location for the converted GT data
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt/kitti/kitti_2d_box'),
            # Location of ground truth data where the seqmap and the clsmap reside
            'SPLIT_TO_CONVERT': 'training',  # Split to convert
            'TRACKER_LIST': None,  # List of trackers to convert, None for all in folder
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
        self.tracker_fol = config['ORIGINAL_TRACKER_FOLDER']
        self.new_tracker_folder = os.path.join(config['NEW_TRACKER_FOLDER'], config['SPLIT_TO_CONVERT'])
        if not config['TRACKER_LIST']:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            tracker_dirs = os.listdir(self.tracker_fol)
            for tracker in self.tracker_list:
                assert tracker in tracker_dirs, 'Tracker directory for tracker %s missing in %s' \
                                                % (tracker, self.tracker_fol)
            self.tracker_list = config['TRACKER_LIST']
        self.gt_dir = config['GT_FOLDER']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
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
            # sequence path
            file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.txt')

            lines = []
            with open(file) as fp:
                dialect = csv.Sniffer().sniff(fp.readline())  # Auto determine file structure.
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    lines.append('%s %s %d %d %d %s %f %f %f %f %f\n'
                                 % (row[0], row[1], self.class_name_to_class_id[row[2].lower()], 0, 0, 'None',
                                    float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[17])))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = Kitti2DBoxTrackerConverter.get_default_config()
    conf = utils.update_config(default_conf)
    Kitti2DBoxTrackerConverter(conf).convert()
