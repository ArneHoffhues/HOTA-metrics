import sys
import os
from glob import glob
from pathlib import Path
import json
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


def get_default_config():
    code_path = utils.get_code_path()
    default_config = {
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/bdd100k/'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/bdd100k/'),  # Location for the converted GT data
        'SPLIT_TO_CONVERT': 'val',  # Split to convert
        'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    print("Converting BDD100K ground truth data...")
    for k, v in config.items():
        print("%s: %s" % (k, v))
    gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], config['SPLIT_TO_CONVERT'])

    class_name_to_class_id = {'pedestrian': 1, 'rider': 2, 'other person': 3, 'car': 4, 'bus': 5, 'truck': 6,
                              'train': 7, 'trailer': 8, 'other vehicle': 9, 'motorcycle': 10, 'bicycle': 11}

    # determine sequences
    sequences = glob(os.path.join(gt_fol, '*.json'))
    seq_list = [seq.split('/')[-1].split('.')[0] for seq in sequences]
    seq_lengths = {}

    # create folder for the new ground truth data if not present
    new_gt_folder = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
    if not config['OUTPUT_AS_ZIP']:
        data_dir = os.path.join(new_gt_folder, 'data')
    else:
        data_dir = os.path.join(new_gt_folder, 'tmp')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # iterate over sequences and write a text file with the annotations for each
    for seq in seq_list:
        # load sequence
        seq_file = os.path.join(gt_fol, seq + '.json')
        with open(seq_file) as f:
            data = json.load(f)
        # order by timestep
        data = sorted(data, key=lambda x: x['index'])
        num_timesteps = len(data)
        seq_lengths[seq] = num_timesteps

        # write to sequence files
        seq_file = os.path.join(data_dir, seq + '.txt')
        lines = []
        for t in range(num_timesteps):
            for label in data[t]['labels']:
                # in bdd100k distractor class annotations and crowd regions are ignore regions
                if 'attributes' in label.keys():
                    is_crowd = int(label['attributes']['Crowd'])
                    is_truncated = int(label['attributes']['Truncated'])
                    is_occluded = int(label['attributes']['Occluded'])
                else:
                    is_crowd = 0
                    is_truncated = 0
                    is_occluded = 0
                lines.append('%d %d %d %d %d %d %d %d %s %d %d %d %d\n'
                             % (t, int(label['id']), class_name_to_class_id[label['category']], is_crowd, is_truncated,
                                is_occluded, 0, 0, 'None', label['box2d']['x1'], label['box2d']['y1'],
                                label['box2d']['x2'], label['box2d']['y2']))
        with open(seq_file, 'w') as f:
            f.writelines(lines)

    # zip the output files and delete temporary data directory
    if config['OUTPUT_AS_ZIP']:
        output_filename = os.path.join(new_gt_folder, 'data')
        shutil.make_archive(output_filename, 'zip', data_dir)
        shutil.rmtree(data_dir)

    # write the class name to class id maps to file
    lines = ['%s %d\n' % (k, v) for k, v in class_name_to_class_id.items()]
    clsmap_file = os.path.join(new_gt_folder, config['SPLIT_TO_CONVERT'] + '.clsmap')

    with open(clsmap_file, 'w') as f:
        f.writelines(lines)

    # write the sequence maps to file
    lines = ['%s empty %d %d\n' % (k, 0, v) for k, v in seq_lengths.items()]
    seqmap_file = os.path.join(new_gt_folder, config['SPLIT_TO_CONVERT'] + '.seqmap')

    with open(seqmap_file, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    config = utils.update_config(get_default_config())
    convert(config)