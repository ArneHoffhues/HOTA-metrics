import sys
import os
from pathlib import Path
import csv
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


def get_default_config():
    code_path = utils.get_code_path()
    default_config = {
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_mots'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/kitti/kitti_mots'),
        # Location for the converted GT data
        'SPLIT_TO_CONVERT': 'val',  # Split to convert
        'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    print("Converting KITTI MOTS ground truth data...")
    for k, v in config.items():
        print("%s: %s" % (k, v))
    gt_fol = config['ORIGINAL_GT_FOLDER']

    class_name_to_class_id = {'cars': 1, 'pedestrians': 2, 'ignore': 10}

    # Get sequences to eval and check gt files exist
    seq_list = []
    seq_lengths = {}
    seqmap_name = config['SPLIT_TO_CONVERT'] + ".seqmap"
    seqmap_file = os.path.join(gt_fol, seqmap_name)
    assert os.path.isfile(seqmap_file), 'no seqmap %s found in %s' % (seqmap_name, gt_fol)

    with open(seqmap_file, "r") as fp:
        dialect = csv.Sniffer().sniff(fp.read(1024))
        fp.seek(0)
        reader = csv.reader(fp, dialect)
        for row in reader:
            if len(row) >= 4:
                seq = "%04d" % int(row[0])
                seq_list.append(seq)
                seq_lengths[seq] = int(row[3]) + 1
                assert os.path.isfile(os.path.join(gt_fol, 'instances_txt', seq + '.txt')), \
                    'GT file %s.txt not found in %s' % (seq, os.path.join(gt_fol, 'instances_txt'))

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
        file = os.path.join(gt_fol, 'instances_txt', seq + '.txt')

        # write to sequence files
        seq_file = os.path.join(data_dir, seq + '.txt')
        lines = []
        with open(file) as fp:
            dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=' ')  # Auto determine file structure.
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for row in reader:
                lines.append('%s %s %s %d %d %d %s %s %s %f %f %f %f\n'
                             % (row[0], row[1], row[2], 0, 0, 0, row[3], row[4], row[5], 0, 0, 0, 0))
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