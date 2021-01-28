import sys
import os
import csv
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


def get_default_config():
    code_path = utils.get_code_path()
    default_config = {
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/davis/'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/davis/'),  # Location for the converted GT data
        'SPLIT_TO_CONVERT': 'val',  # Split to convert
        'OUTPUT_AS_ZIP': True  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], 'Annotations_unsupervised/480p')

    # check if all gt data is present and determine sequence lengths
    seq_list = []
    seq_lengths = {}
    seqmap_file = os.path.join(config['ORIGINAL_GT_FOLDER'], 'ImageSets/2017', config['SPLIT_TO_CONVERT'] + '.txt')
    assert os.path.isfile(seqmap_file), 'no seqmap found: ' + seqmap_file
    with open(seqmap_file) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if row[0] == '':
                continue
            seq = row[0]
            seq_list.append(seq)
            curr_dir = os.path.join(gt_fol, seq)
            assert os.path.isdir(curr_dir), 'GT directory not found: ' + curr_dir
            curr_timesteps = len(glob(os.path.join(curr_dir, '*.png')))
            seq_lengths[seq] = curr_timesteps

    # write sequence map data to file
    lines = ['%s empty %d %d\n' % (k, 0, v) for k, v in seq_lengths.items()]
    seqmap_file = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'],
                               config['SPLIT_TO_CONVERT'] + '.seqmap')

    with open(seqmap_file, 'w') as f:
        f.writelines(lines)

    # set data output directory
    if not config['OUTPUT_AS_ZIP']:
        data_dir = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'], 'data')
    else:
        data_dir = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'], 'tmp')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # iterate over sequences and write a text file with the annotations for each
    for seq in tqdm(seq_list, desc='Converting DAVIS ground truth data'):
        seq_dir = os.path.join(gt_fol, seq)
        num_timesteps = seq_lengths[seq]
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

        # write to sequence file
        seq_file = os.path.join(data_dir, seq + '.txt')
        lines = []
        for t in range(num_timesteps):
            to_append = ['%d %d %d %d %d %s\n' % (t, i, 0, masks_encoded[i][t]['size'][0],
                                                  masks_encoded[i][t]['size'][1], masks_encoded[i][t]['counts'])
                         for i in masks_encoded.keys()]
            lines += to_append
            lines += ['%d %d %d %d %d %s\n' % (t, -1, 1, masks_void[t]['size'][0],
                                               masks_void[t]['size'][1], masks_void[t]['counts'])]
        with open(seq_file, 'w') as f:
            f.writelines(lines)

    # zip directory and delete temporary data directory
    if config['OUTPUT_AS_ZIP']:
        output_filename = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'], 'data')
        shutil.make_archive(output_filename, 'zip', data_dir)
        shutil.rmtree(data_dir)


if __name__ == '__main__':
    config = utils.update_config(get_default_config())
    convert(config)

