import sys
import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class DAVISConverter(_BaseDatasetConverter):
    """Converter for Davis ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/davis/'),
            # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/davis/'),  # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'val',  # Split to convert
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
        self.gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], 'davis_unsupervised_' + config['SPLIT_TO_CONVERT'])
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = 'davis_unsupervised_' + config['SPLIT_TO_CONVERT']
        self.class_name_to_class_id = {'general': 1, 'void': 2}

        # Get sequences to convert and check gt files exist
        self.seq_list = os.listdir(self.gt_fol)
        self.seq_lengths = {seq: len(os.listdir(os.path.join(self.gt_fol, seq))) for seq in self.seq_list}
        self.seq_sizes = {}


    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        # import for reduction of minimum requirements
        from pycocotools.mask import encode

        data = {}
        for seq in tqdm(self.seq_list):
            seq_dir = os.path.join(self.gt_fol, seq)
            num_timesteps = self.seq_lengths[seq]
            frames = np.sort(glob(os.path.join(seq_dir, '*.png')))

            # open ground truth masks
            mask0 = np.array(Image.open(frames[0]))
            self.seq_sizes[seq] = mask0.shape

            lines = []
            for t in range(num_timesteps):
                frame = np.array(Image.open(frames[t]))
                void = frame == 255
                frame[void] = 0
                void_encoded = encode(np.asfortranarray(void.astype(np.uint8)))
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
                lines += ['%d %d %d %d %d %s %f %f %f %f %d %d %d %d \n'
                          % (t, -1, 2, void_encoded['size'][0], void_encoded['size'][1],
                             void_encoded['counts'].decode("utf-8"), 0, 0, 0, 0, 0, 0, 0, 0)]
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = DAVISConverter.get_default_config()
    conf = utils.update_config(default_conf)
    DAVISConverter(conf).convert()
