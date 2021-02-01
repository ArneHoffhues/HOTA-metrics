import sys
import os
from glob import glob
from pathlib import Path
import json
from pycocotools import mask as mask_utils
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


def get_default_config():
    code_path = utils.get_code_path()
    default_config = {
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/youtube_vis/'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/youtube_vis/'),  # Location for the converted GT data
        'SPLIT_TO_CONVERT': 'training',  # Split to convert
        'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    print("Converting YouTubeVIS ground truth data...")
    for k, v in config.items():
        print("%s: %s" % (k, v))
    gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], config['SPLIT_TO_CONVERT'])

    gt_dir_files = glob(os.path.join(gt_fol, '*.json'))
    assert len(gt_dir_files) == 1, gt_fol + ' does not contain exactly one json file.'

    with open(gt_dir_files[0]) as f:
        data = json.load(f)

    class_name_to_class_id = {cat['name']: cat['id'] for cat in data['categories']}

    # determine sequences
    sequences = {vid['file_names'][0].split('/')[0]: vid['id'] for vid in data['videos']}
    seq_list = list(sequences.keys())
    seq_lengths = {vid['file_names'][0].split('/')[0]: vid['length'] for vid in data['videos']}

    # create folder for the new ground truth data if not present
    new_gt_folder = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
    Path(new_gt_folder).mkdir(parents=True, exist_ok=True)
    if not config['OUTPUT_AS_ZIP']:
        data_dir = os.path.join(new_gt_folder, 'data')
    else:
        data_dir = os.path.join(new_gt_folder, 'tmp')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # iterate over sequences and write a text file with the annotations for each
    for seq in seq_list:
        # determine annotations for given sequence
        seq_id = sequences[seq]
        seq_annotations = [ann for ann in data['annotations'] if ann['video_id'] == seq_id]

        # write to sequence files
        seq_file = os.path.join(data_dir, seq + '.txt')
        lines = []
        for t in range(seq_lengths[seq]):
            for ann in seq_annotations:
                if ann['segmentations'][t]:
                    h = ann['height']
                    w = ann['width']
                    mask_encoded = mask_utils.frPyObjects(ann['segmentations'][t], h, w)
                    lines.append('%d %d %d %d %d %d %d %d %s %d %d %d %d\n'
                                 % (t, ann['id'], ann['category_id'], ann['iscrowd'], 0, 0, h, w,
                                    mask_encoded['counts'], 0, 0, 0, 0))
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
    lines = ['%s %d\n' % (k, v) for k, v in seq_lengths.items()]
    seqmap_file = os.path.join(new_gt_folder, config['SPLIT_TO_CONVERT'] + '.seqmap')

    with open(seqmap_file, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    config = utils.update_config(get_default_config())
    convert(config)