import sys
import os
from pathlib import Path
import csv
import shutil
import configparser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


def get_default_config():
    code_path = utils.get_code_path()
    default_config = {
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/mot_challenge/mot_challenge_2d_box'),
        # Location for the converted GT data
        'BENCHMARK': 'MOT17',  # Benchmark to convert
        'SPLIT_TO_CONVERT': 'train',  # Split to convert
        'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    print("Converting MOT Challenge 2D Box ground truth data...")
    for k, v in config.items():
        print("%s: %s" % (k, v))
    gt_set = config['BENCHMARK'] + '-' + config['SPLIT_TO_CONVERT']
    gt_fol = config['ORIGINAL_GT_FOLDER'] + gt_set

    class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                              'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                              'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}

    # Get sequences to eval and check gt files exist
    seq_list = []
    seq_lengths = {}
    seqmap_file = os.path.join(config['ORIGINAL_GT_FOLDER'], 'seqmaps', gt_set + '.txt')
    if not os.path.isfile(seqmap_file):
        raise Exception('no seqmap found: ' + os.path.basename(seqmap_file))
    with open(seqmap_file) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i == 0 or row[0] == '':
                continue
            seq = row[0]
            seq_list.append(seq)
            ini_file = os.path.join(gt_fol, seq, 'seqinfo.ini')
            if not os.path.isfile(ini_file):
                raise Exception('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
            ini_data = configparser.ConfigParser()
            ini_data.read(ini_file)
            seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
            curr_file = os.path.join(gt_fol, seq, 'gt', 'gt.txt')
            if not os.path.isfile(curr_file):
                raise Exception('GT file not found: ' + seq + '/gt/' + os.path.basename(curr_file))

    # create folder for the new ground truth data if not present
    new_gt_folder = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
    if not config['OUTPUT_AS_ZIP']:
        data_dir = os.path.join(new_gt_folder, 'data')
    else:
        data_dir = os.path.join(new_gt_folder, 'tmp')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # iterate over sequences and write a text file with the annotations for each
    for seq in seq_list:
        # sequence path
        file = os.path.join(gt_fol, seq, 'gt', 'gt.txt')

        # write to sequence files
        seq_file = os.path.join(data_dir, seq + '.txt')
        lines = []
        with open(file) as fp:
            dialect = csv.Sniffer().sniff(fp.read(10240), delimiters=',')  # Auto determine file structure.
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            # convert box format from xywh to x0y0x1y1
            for row in reader:
                lines.append('%s %s %s %d %d %d %s %d %d %s %f %f %f %f\n'
                             % (row[0], row[1], row[7], 0, 0, 0, row[6], 0, 0, 'None', float(row[2]), float(row[3]),
                                float(row[2]) + float(row[4]), float(row[3]) + float(row[5])))
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
