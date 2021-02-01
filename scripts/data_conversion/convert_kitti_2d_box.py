import sys
import os
from glob import glob
from pathlib import Path
import csv
from pycocotools import mask as mask_utils
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


def get_default_config():
    code_path = utils.get_code_path()
    default_config = {
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/kitti/kitti_2d_box'),  # Location for the converted GT data
        'SPLIT_TO_CONVERT': 'val',  # Split to convert
        'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    print("Converting KITTI 2D Box ground truth data...")
    for k, v in config.items():
        print("%s: %s" % (k, v))
    gt_fol = config['ORIGINAL_GT_FOLDER']

    class_name_to_class_id = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person': 5,
                              'Cyclist': 6, 'Tram': 7, 'Misc': 8, 'DontCare': 9}

    # Get sequences to eval and check gt files exist
    seq_list = []
    seq_lengths = {}
    seqmap_name = 'evaluate_tracking.seqmap.' + config['SPLIT_TO_CONVERT']
    seqmap_file = os.path.join(gt_fol, seqmap_name)
    if not os.path.isfile(seqmap_file):
        raise Exception('no seqmap found: ' + os.path.basename(seqmap_file))
    with open(seqmap_file) as fp:
        dialect = csv.Sniffer().sniff(fp.read(1024))
        fp.seek(0)
        reader = csv.reader(fp, dialect)
        for row in reader:
            if len(row) >= 4:
                seq = row[0]
                seq_list.append(seq)
                seq_lengths[seq] = int(row[3])
                curr_file = os.path.join(gt_fol, 'label_02', seq + '.txt')
                if not os.path.isfile(curr_file):
                    raise Exception('GT file not found: ' + os.path.basename(curr_file))

    # create folder for the new ground truth data if not present
    new_gt_folder = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
    if not config['OUTPUT_AS_ZIP']:
        data_dir = os.path.join(new_gt_folder, 'data')
    else:
        data_dir = os.path.join(new_gt_folder, 'tmp')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # iterate over sequences and write a text file with the annotations for each
    for seq in seq_list:
        # load sequences
        file = os.path.join(gt_fol, 'label_02', seq + '.txt')

        # write to sequence files
        seq_file = os.path.join(data_dir, seq + '.txt')
        lines = []
        with open(file) as fp:
            dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=' ')  # Auto determine file structure.
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for row in reader:
                lines.append('%s %s %d %d %s %s %d %d %s %s %s %s %s\n'
                             % (row[0], row[1], class_name_to_class_id[row[2]], 0, row[3], row[4], 0, 0, 'None',
                                row[6], row[7], row[8], row[9]))
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