from convert_bdd100k_2d_box import BDD100K2DBoxConverter
from convert_bdd100k_mots import BDD100KMOTSConverter
from convert_tao import TAOConverter
from convert_youtube_vis import YouTubeVISConverter
from convert_davis import DAVISConverter
from convert_mots_challenge import MOTSChallengeConverter
from convert_mot_2d_box import MOTChallenge2DBoxConverter
from convert_kitti_2d_box import Kitti2DBoxConverter
from convert_kitti_mots import KittiMOTSConverter
from convert_ovis import OVISConverter
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils # noqa: E402

if __name__ == '__main__':
    dataset_converter = [BDD100K2DBoxConverter, BDD100KMOTSConverter, TAOConverter, YouTubeVISConverter, OVISConverter,
                         DAVISConverter, MOTSChallengeConverter, MOTChallenge2DBoxConverter, Kitti2DBoxConverter,
                         KittiMOTSConverter]
    combined_conf = {'OUTPUT_AS_ZIP': False}
    for converter in dataset_converter:
        default_conf = converter.get_default_config()
        default_conf.pop('OUTPUT_AS_ZIP', None)
        dataset_name = converter.get_dataset_name()
        for key, value in default_conf.items():
            combined_conf[dataset_name + '_' + key] = value
    conf = utils.update_config(combined_conf)

    for converter in dataset_converter:
        dataset_name = converter.get_dataset_name()
        conv_config = {k.split('_', maxsplit=1)[-1]: v for k, v in combined_conf.items() if k.startswith(dataset_name)}
        conv_config['OUTPUT_AS_ZIP'] = combined_conf['OUTPUT_AS_ZIP']
        converter(conv_config).convert()
