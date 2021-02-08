import sys
import os
import csv
from glob import glob
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
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
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/davis/'),  # Location of original GT data
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
        self.gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], 'Annotations_unsupervised/480p')
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']
        self.class_name_to_class_id = {'general': 1, 'void': 2}

        # Get sequences to convert and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        seqmap_file = os.path.join(config['ORIGINAL_GT_FOLDER'], 'ImageSets/2017', config['SPLIT_TO_CONVERT'] + '.txt')
        assert os.path.isfile(seqmap_file), 'no seqmap found: ' + seqmap_file
        with open(seqmap_file) as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if row[0] == '':
                    continue
                seq = row[0]
                self.seq_list.append(seq)
                curr_dir = os.path.join(self.gt_fol, seq)
                assert os.path.isdir(curr_dir), 'GT directory not found: ' + curr_dir
                curr_timesteps = len(glob(os.path.join(curr_dir, '*.png')))
                self.seq_lengths[seq] = curr_timesteps

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        data = {}
        for seq in tqdm(self.seq_list):
            seq_dir = os.path.join(self.gt_fol, seq)
            num_timesteps = self.seq_lengths[seq]
            frames = np.sort(glob(os.path.join(seq_dir, '*.png')))

            # open ground truth masks
            mask0 = np.array(Image.open(frames[0]))
            all_masks = np.zeros((num_timesteps, *mask0.shape))
            for i, t in enumerate(frames):
                all_masks[i, ...] = np.array(Image.open(t))

            # determine and encode void masks
            masks_void = all_masks == 255
            masks_void = mask_utils.encode(np.array(np.transpose(masks_void, (1, 2, 0)), order='F').astype(np.uint8))

            # split tracks and encode them
            num_objects = int(np.max(all_masks))
            tmp = np.ones((num_objects, *all_masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = np.array(tmp == all_masks[None, ...]).astype(np.uint8)
            masks_encoded = {i: mask_utils.encode(np.array(
                np.transpose(masks[i, :], (1, 2, 0)), order='F')) for i in range(masks.shape[0])}

            lines = []
            for t in range(num_timesteps):
                to_append = ['%d %d %d %d %d %d %d %d %d %s %f %f %f %f\n'
                             % (t, i, 1, 0, 0, 0, 0, masks_encoded[i][t]['size'][0], masks_encoded[i][t]['size'][1],
                                masks_encoded[i][t]['counts'], 0, 0, 0, 0)
                             for i in masks_encoded.keys()]
                lines += to_append
                lines += ['%d %d %d %d %d %d %d %d %d %s %f %f %f %f\n'
                          % (t, -1, 2, 0, 0, 0, 0, masks_void[t]['size'][0], masks_void[t]['size'][1],
                             masks_void[t]['counts'], 0, 0, 0, 0)]
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = DAVISConverter.get_default_config()
    conf = utils.update_config(default_conf)
    DAVISConverter(conf).convert()
