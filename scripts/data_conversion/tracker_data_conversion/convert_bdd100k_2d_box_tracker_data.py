import sys
import os
import json
from tqdm import tqdm
from _base_tracker_data_converter import _BaseTrackerDataConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class BDD100KTrackerConverter(_BaseTrackerDataConverter):
    """Converter for BDD100K tracker data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_TRACKER_FOLDER': os.path.join(code_path, 'data/trackers/bdd100k/'),
            # Location of original GT data
            'NEW_TRACKER_FOLDER': os.path.join(code_path, 'data/converted_trackers/bdd100k/'),
            # Location for the converted GT data
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt/bdd100k/'),
            # Location of ground truth data where the seqmap and the clsmap reside
            'SPLIT_TO_CONVERT': 'val',  # Split to convert
            'TRACKER_LIST': None,  # List of trackers to convert, None for all in folder
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
        self.split_to_convert = 'bdd100k_2d_box_' + config['SPLIT_TO_CONVERT']
        self.tracker_fol = os.path.join(config['ORIGINAL_TRACKER_FOLDER'], self.split_to_convert)
        self.new_tracker_folder = os.path.join(config['NEW_TRACKER_FOLDER'], self.split_to_convert)
        if not config['TRACKER_LIST']:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            tracker_dirs = os.listdir(self.tracker_fol)
            for tracker in config['TRACKER_LIST']:
                assert tracker in tracker_dirs, 'Tracker directory for tracker %s missing in %s' \
                                                % (tracker, self.tracker_fol)
            self.tracker_list = config['TRACKER_LIST']
        self.gt_dir = config['GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']

        # Get sequences
        self._get_sequences()

        # check if all tracker files are present
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.json')
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
        for seq in tqdm(self.seq_list):
            # load sequence
            seq_file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.json')
            with open(seq_file) as f:
                tracker_data = json.load(f)

            # order by timestep
            tracker_data = sorted(tracker_data, key=lambda x: x['index'])
            num_timesteps = self.seq_lengths[seq]

            lines = []
            for t in range(num_timesteps):
                for label in tracker_data[t]['labels']:
                    # compute smallest mask which fully covers the bounding box
                    # mask = np.zeros(self.seq_sizes[seq], order='F').astype(np.uint8)
                    # mask[int(np.floor(label['box2d']['y1'])):int(np.ceil(label['box2d']['y2'])+1),
                    #      int(np.floor(label['box2d']['x1'])):int(np.ceil(label['box2d']['x2'])+1)] = 1
                    # encoded_mask = mask_utils.encode(mask)
                    # convert bbox data into x0y0x1y1 format
                    # lines.append('%d %d %d %d %d %s %.13f %.13f %.13f %.13f %.15f\n'
                    #              % (t, int(label['id']), self.class_name_to_class_id[label['category']],
                    #                 encoded_mask['size'][0], encoded_mask['size'][1], encoded_mask['counts'],
                    #                 label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'],
                    #                 label['box2d']['y2'], label['score']))
                    lines.append('%d %d %d %d %d %s %.13f %.13f %.13f %.13f %.15f\n'
                                 % (t, int(label['id']), self.class_name_to_class_id[label['category']],
                                    0, 0, 'None', label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'],
                                    label['box2d']['y2'], label['score']))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = BDD100KTrackerConverter.get_default_config()
    conf = utils.update_config(default_conf)
    BDD100KTrackerConverter(conf).convert()
