import sys
import os
from _base_tracker_data_converter import _BaseTrackerDataConverter
from tqdm import tqdm
import numpy as np
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class DAVISTrackerConverter(_BaseTrackerDataConverter):
    """Converter for DAVIS tracker data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_TRACKER_FOLDER': os.path.join(code_path, 'data/trackers/davis/'),
            # Location of original GT data
            'NEW_TRACKER_FOLDER': os.path.join(code_path, 'data/converted_trackers/davis'),
            # Location for the converted GT data
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt/davis'),
            # Location of ground truth data where the seqmap and the clsmap reside
            'SPLIT_TO_CONVERT': 'val',  # Split to convert
            'TRACKER_LIST': None,  # List of trackers to convert, None for all in folder
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'DAVIS'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split_to_convert = 'davis_unsupervised_' + config['SPLIT_TO_CONVERT']
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
                curr_seq = os.path.join(self.tracker_fol, tracker, 'data', seq)
                if not os.path.isdir(curr_seq):
                    raise Exception('Tracker sequence %s not found for tracker %s.'
                                    % (curr_seq, tracker))
                if not len(os.listdir(curr_seq)) == self.seq_lengths[seq]:
                    raise Exception('Sequence length  and tracker data do not match  for sequence %s and tracker %s. '
                                    'Found %d timesteps instead of %d'
                                    % (seq, tracker, len(os.listdir(curr_seq)), self.seq_lengths[seq]))

        # Get classes
        self._get_classes()

    def _prepare_data(self, tracker):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        # import for reduction of minimum requirements
        from pycocotools.mask import encode
        from PIL import Image

        data = {}
        for seq in tqdm(self.seq_list):
            seq_dir = os.path.join(self.tracker_fol, tracker, 'data', seq)
            num_timesteps = self.seq_lengths[seq]
            frames = np.sort(glob(os.path.join(seq_dir, '*.png')))

            # # load mask images
            # all_masks = np.zeros((num_timesteps, *self.seq_sizes[seq]))
            # for i, t in enumerate(frames):
            #     all_masks[i, ...] = np.array(Image.open(t))
            #
            # # split tracks and encode them
            # num_objects = int(np.max(all_masks))
            # tmp = np.ones((num_objects, *all_masks.shape))
            # tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            # masks = np.array(tmp == all_masks[None, ...]).astype(np.uint8)
            # masks_encoded = {i: encode(np.array(np.transpose(masks[i, :], (1, 2, 0)), order='F'))
            #                  for i in range(masks.shape[0])}

            lines = []
            for t in range(num_timesteps):
                frame = np.array(Image.open(frames[t]))
                id_values = np.unique(frame)
                id_values = id_values[id_values != 0]
                tmp = np.ones((len(id_values), *frame.shape))
                tmp = tmp * id_values[:, None, None]
                masks = np.array(tmp == frame[None, ...]).astype(np.uint8)
                masks_encoded = encode(np.array(np.transpose(masks, (1, 2, 0)), order='F'))
                ids = id_values.astype(int)

                to_append = ['%d %d %d %d %d %s %f %f %f %f %d %d %d %d \n'
                             % (t, ids[i], 1, masks_encoded[i]['size'][0], masks_encoded[i]['size'][1],
                                masks_encoded[i]['counts'].decode("utf-8"), 0, 0, 0, 0, 0, 0, 0, 0)
                             for i in range(len(masks_encoded))]
                lines += to_append
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = DAVISTrackerConverter.get_default_config()
    conf = utils.update_config(default_conf)
    DAVISTrackerConverter(conf).convert()