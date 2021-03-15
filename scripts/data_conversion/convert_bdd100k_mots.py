import sys
import os
import numpy as np
from _base_dataset_converter import _BaseDatasetConverter
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class BDD100KMOTSConverter(_BaseDatasetConverter):
    """Converter for BDDMOTS ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/bdd100k/'),
            # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/bdd100k/'),
            # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'val',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'BDD100KMOTS'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = config['ORIGINAL_GT_FOLDER'] + 'bdd100k_mots_' + config['SPLIT_TO_CONVERT']
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = 'bdd100k_mots_' + config['SPLIT_TO_CONVERT']

        self.class_name_to_class_id = {'pedestrian': 1, 'rider': 2, 'car': 3, 'truck': 4, 'bus': 5, 'train': 6,
                                       'motorcycle': 7, 'bicycle': 8}

        # Get sequences
        self.seq_list = [seq_file for seq_file in os.listdir(os.path.join(self.gt_fol, 'bitmasks'))]
        self.seq_lengths = {seq: len(os.listdir(os.path.join(self.gt_fol, 'bitmasks', seq))) for seq in self.seq_list}
        self.seq_sizes = {}

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        from PIL import Image
        from pycocotools import mask as mask_utils

        data = {}
        for seq in tqdm(self.seq_list):
            # get sequence masks
            bitmask_folder = os.path.join(self.gt_fol, 'bitmasks', seq)
            bitmasks = sorted(os.listdir(bitmask_folder))

            num_timesteps = self.seq_lengths[seq]

            lines = []
            for t in range(num_timesteps):
                # load bitmask for timestep
                bitmask_file = os.path.join(bitmask_folder, bitmasks[t])
                bitmask = np.asarray(Image.open(bitmask_file)).astype(np.int32)
                if seq not in self.seq_sizes:
                    self.seq_sizes[seq] = bitmask.shape[:2]

                instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]
                category_map = bitmask[:, :, 0]
                attributes_map = bitmask[:, :, 1]
                instance_ids = np.sort(np.unique(instance_map[instance_map >= 1]))

                for i, instance_id in enumerate(instance_ids):
                    mask_inds_i = instance_map == instance_id
                    mask_encoded = mask_utils.encode(np.asfortranarray(mask_inds_i).astype(np.uint8))
                    attributes_i = np.unique(attributes_map[mask_inds_i])
                    assert attributes_i.shape[0] == 1
                    # attributes:   first bit - ignore, second bit - crowd, third bit - occluded,
                    #               fourth bit - truncated
                    ignore_i = attributes_i[0] & 1
                    crowd_i = attributes_i[0] >> 1 & 1
                    occluded_i = attributes_i[0] >> 2 & 1
                    truncated_i = attributes_i[0] >> 3 & 1
                    category_ids_i = np.unique(category_map[mask_inds_i])
                    assert category_ids_i.shape[0] == 1

                    lines.append('%d %d %d %d %d %s %.13f %.13f %.13f %.13f %d %d %d %d\n'
                                 % (t, int(instance_id), category_ids_i[0], mask_encoded['size'][0],
                                    mask_encoded['size'][1], mask_encoded['counts'].decode("utf-8"),
                                    0, 0, 0, 0, crowd_i, truncated_i, occluded_i, ignore_i))
            data[seq] = lines
        return data


if __name__ == '__main__':
    default_conf = BDD100KMOTSConverter.get_default_config()
    conf = utils.update_config(default_conf)
    BDD100KMOTSConverter(conf).convert()