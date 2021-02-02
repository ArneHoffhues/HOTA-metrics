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
        'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/tao/'),  # Location of original GT data
        'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/tao/'),  # Location for the converted GT data
        'SPLIT_TO_CONVERT': 'training',  # Split to convert
        'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
    }
    return default_config


def convert(config):
    print("Converting TAO ground truth data...")
    for k, v in config.items():
        print("%s: %s" % (k, v))
    gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], config['SPLIT_TO_CONVERT'])

    gt_dir_files = glob(os.path.join(gt_fol, '*.json'))
    assert len(gt_dir_files) == 1, gt_fol + ' does not contain exactly one json file.'

    with open(gt_dir_files[0]) as f:
        data = json.load(f)

    class_name_to_class_id = {cat['name']: cat['id'] for cat in data['categories']}
    merge_map = {}
    for category in data['categories']:
        if 'merged' in category:
            for to_merge in category['merged']:
                merge_map[to_merge['id']] = category['id']
    merge_map = {cat_id: merge_map[cat_id] if cat_id in merge_map.keys() else cat_id for cat_id
                 in [cat['id'] for cat in data['categories']]}

    # determine sequences
    sequences = {vid['name'].replace('/', '-'): vid['id'] for vid in data['videos']}
    seq_list = list(sequences.keys())
    seq_properties = {vid['id']: {'pos_category_ids': set(),
                                  'neg_category_ids': vid['neg_category_ids'],
                                  'not_exhaustively_labeled_ids': vid['not_exhaustive_category_ids']}
                      for vid in data['videos']}

    # compute mapping from videos to images to determine seqeuence lengths and sort images by occurence
    vids_to_images = {vid['id']: [] for vid in data['videos']}
    for img in data['images']:
        vids_to_images[img['video_id']].append(img)
    vids_to_images = {k: sorted(v, key=lambda x: x['frame_index']) for k, v in vids_to_images.items()}
    for vid, imgs in vids_to_images.items():
        seq_properties[vid]['length'] = len(imgs)

    # create folder for the new ground truth data if not present
    new_gt_folder = os.path.join(config['NEW_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
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
        for t in range(seq_properties[seq_id]['length']):
            # determine annotations for given timestep
            timestep_annotations = [ann for ann in seq_annotations
                                    if ann['image_id'] == vids_to_images[seq_id][t]['id']]
            for ann in timestep_annotations:
                seq_properties[seq_id]['pos_category_ids'].add(ann['category_id'])
                # convert box format from xywh to x0y0x1y1
                lines.append('%d %d %d %d %d %d %d %d %d %s %f %f %f %f\n'
                             % (t, ann['id'], ann['category_id'], ann['iscrowd'], 0, 0, 0, 0, 0, 'None',
                                ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                                ann['bbox'][1] + ann['bbox'][3]))
        with open(seq_file, 'w') as f:
            f.writelines(lines)

    # zip the output files and delete temporary data directory
    if config['OUTPUT_AS_ZIP']:
        output_filename = os.path.join(new_gt_folder, 'data')
        shutil.make_archive(output_filename, 'zip', data_dir)
        shutil.rmtree(data_dir)

    # write the class name to class id maps to file
    lines = ['%s %d %d\n' % (k, v, merge_map[v]) for k, v in class_name_to_class_id.items()]
    clsmap_file = os.path.join(new_gt_folder, config['SPLIT_TO_CONVERT'] + '.clsmap')

    with open(clsmap_file, 'w') as f:
        f.writelines(lines)

    # write the sequence maps to file
    for vid in seq_properties.keys():
        seq_properties[vid]['pos_category_ids'] = list(seq_properties[vid]['pos_category_ids'])
    lines = ['%s %d %s %s %s\n' % (k, seq_properties[v]['length'],
                                   ','.join(str(id) for id in seq_properties[v]['pos_category_ids']),
                                   ','.join(str(id) for id in seq_properties[v]['neg_category_ids']),
                                   ','.join(str(id) for id in seq_properties[v]['not_exhaustively_labeled_ids']))
             for k, v in sequences.items()]
    seqmap_file = os.path.join(new_gt_folder, config['SPLIT_TO_CONVERT'] + '.seqmap')

    with open(seqmap_file, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    config = utils.update_config(get_default_config())
    convert(config)