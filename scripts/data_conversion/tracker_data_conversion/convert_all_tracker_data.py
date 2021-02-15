from convert_bdd100k_tracker_data import BDD100KTrackerConverter
from convert_davis_tracker_data import DAVISTrackerConverter
from convert_kitti_2d_box_tracker_data import Kitti2DBoxTrackerConverter
from convert_kitti_mots_tracker_data import KittiMOTSTrackerConverter
from convert_mot_2d_box_tracker_data import MOTChallenge2DBoxTrackerConverter
from convert_mots_challenge_tracker_data import MOTSChallengeTrackerConverter
from convert_tao_tracker_data import TAOTrackerConverter
from convert_youtube_vis_tracker_data import YouTubeVISTrackerConverter

if __name__ == '__main__':
    tracker_converter = [BDD100KTrackerConverter, TAOTrackerConverter, YouTubeVISTrackerConverter,
                         DAVISTrackerConverter, MOTSChallengeTrackerConverter, MOTChallenge2DBoxTrackerConverter,
                         Kitti2DBoxTrackerConverter, KittiMOTSTrackerConverter]
    for converter in tracker_converter:
        default_conf = converter.get_default_config()
        converter(default_conf).convert()
