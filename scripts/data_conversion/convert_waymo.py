import sys
import os
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from trackeval import utils  # noqa: E402


class WaymoConverter(_BaseDatasetConverter):
    """Converter for TAO ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/waymo/'),  # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/waymo/'),  # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'training',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'Waymo'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], 'waymo_' + config['SPLIT_TO_CONVERT'])
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = 'waymo_' + config['SPLIT_TO_CONVERT']

        # class list and corresponding class ids
        self.class_name_to_class_id = {'TYPE_UNKNOWN': 0, 'TYPE_VEHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 3,
                                       'TYPE_CYCLIST': 4}

        # Get sequences
        self.camera_types = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
        self.seq_list = [str(seq_file).replace('_with_camera_labels.tfrecord', '')
                         for seq_file in os.listdir(self.gt_fol)]
        self.seq_lengths = {}
        self.seq_sizes = {}

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
                 sequence file
        """
        # reduction of minimum requirements
        import tensorflow.compat.v1 as tf
        from tqdm import tqdm
        from waymo_open_dataset import dataset_pb2 as open_dataset
        tf.enable_eager_execution()

        data = {}
        for seq in tqdm(self.seq_list):
            got_metadata = False
            timestep = 0
            track_id_counter = 0
            track_id_mapping = {}
            # load sequence
            seq_file = os.path.join(self.gt_fol, seq + '_with_camera_labels.tfrecord')
            dataset = tf.data.TFRecordDataset(seq_file, compression_type='')
            lines = {'FRONT': [], 'FRONT_LEFT': [], 'FRONT_RIGHT': [], 'SIDE_LEFT': [], 'SIDE_RIGHT': []}
            for frame_data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(frame_data.numpy()))
                if not got_metadata:
                    for camera_cal in frame.context.camera_calibrations:
                        camera_name = self.camera_types[camera_cal.name]
                        self.seq_sizes[seq + '_' + camera_name] = (camera_cal.height, camera_cal.width)
                    got_metadata = True
                for camera in frame.camera_labels:
                    camera_name = self.camera_types[camera.name]
                    for label in camera.labels:
                        if not label.id in track_id_mapping:
                            track_id_mapping[label.id] = track_id_counter
                            track_id_counter += 1
                        lines[camera_name].append('%d %d %d %d %d %s %.13f %.13f %.13f %.13f %d %d %d %d\n'
                                                  % (timestep, track_id_mapping[label.id], label.type, 0, 0, 'None',
                                                     label.box.center_x - 0.5 * label.box.length,
                                                     label.box.center_y - 0.5 * label.box.width,
                                                     label.box.center_x + 0.5 * label.box.length,
                                                     label.box.center_y + 0.5 * label.box.width,
                                                     0, 0, 0, 0))
                timestep += 1
            for camera_name in self.camera_types.values():
                self.seq_lengths[seq + '_' + camera_name] = timestep
            for key, value in lines.items():
                data[seq + '_' + key] = value
        return data


if __name__ == '__main__':
    default_conf = WaymoConverter.get_default_config()
    conf = utils.update_config(default_conf)
    WaymoConverter(conf).convert()
