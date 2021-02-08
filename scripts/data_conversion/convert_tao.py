import sys
import os
from glob import glob
import json
from _base_dataset_converter import _BaseDatasetConverter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from hota_metrics import utils  # noqa: E402


class TAOConverter(_BaseDatasetConverter):
    """Converter for TAO ground truth data"""

    @staticmethod
    def get_default_config():
        """Default converter config values"""
        code_path = utils.get_code_path()
        default_config = {
            'ORIGINAL_GT_FOLDER': os.path.join(code_path, 'data/gt/tao/'),  # Location of original GT data
            'NEW_GT_FOLDER': os.path.join(code_path, 'data/converted_gt/tao/'),  # Location for the converted GT data
            'SPLIT_TO_CONVERT': 'training',  # Split to convert
            'OUTPUT_AS_ZIP': False  # Whether the converted output should be zip compressed
        }
        return default_config

    @staticmethod
    def get_dataset_name():
        """Returns the name of the associated dataset"""
        return 'TAO'

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gt_fol = os.path.join(config['ORIGINAL_GT_FOLDER'], config['SPLIT_TO_CONVERT'])
        self.new_gt_folder = config['NEW_GT_FOLDER']
        self.output_as_zip = config['OUTPUT_AS_ZIP']
        self.split_to_convert = config['SPLIT_TO_CONVERT']

        # read gt file
        gt_dir_files = glob(os.path.join(self.gt_fol, '*.json'))
        assert len(gt_dir_files) == 1, self.gt_fol + ' does not contain exactly one json file.'

        with open(gt_dir_files[0]) as f:
            self.gt_data = json.load(f)

        # class list and corresponding class ids
        self.class_name_to_class_id = {cat['name']: cat['id'] for cat in self.gt_data['categories']}
        # compute categories which should be merged
        merge_map = {}
        for category in self.gt_data['categories']:
            if 'merged' in category:
                for to_merge in category['merged']:
                    merge_map[to_merge['id']] = category['id']
        self.merge_map = {cat_id: merge_map[cat_id] if cat_id in merge_map.keys() else cat_id for cat_id
                          in [cat['id'] for cat in self.gt_data['categories']]}

        # determine sequences
        self.sequences = {vid['name'].replace('/', '-'): vid['id'] for vid in self.gt_data['videos']}
        self.seq_list = list(self.sequences.keys())
        self.seq_properties = {vid['id']: {'pos_category_ids': set(),
                                           'neg_category_ids': vid['neg_category_ids'],
                                           'not_exhaustively_labeled_ids': vid['not_exhaustive_category_ids']}
                               for vid in self.gt_data['videos']}

        # compute mapping from videos to images to determine seqeuence lengths and sort images by occurence
        vids_to_images = {vid['id']: [] for vid in self.gt_data['videos']}
        for img in self.gt_data['images']:
            vids_to_images[img['video_id']].append(img)
        self.vids_to_images = {k: sorted(v, key=lambda x: x['frame_index']) for k, v in vids_to_images.items()}
        for vid, imgs in vids_to_images.items():
            self.seq_properties[vid]['length'] = len(imgs)

    def _prepare_data(self):
        """
        Computes the lines to write for each respective sequence file.
        :return: a dictionary which maps the sequence names to the lines that should be written to the according
             sequence file
        """
        data = {}
        for seq in self.seq_list:
            # determine annotations for given sequence
            seq_id = self.sequences[seq]
            seq_annotations = [ann for ann in self.gt_data['annotations'] if ann['video_id'] == seq_id]

            lines = []
            for t in range(self.seq_properties[seq_id]['length']):
                # determine annotations for given timestep
                timestep_annotations = [ann for ann in seq_annotations
                                        if ann['image_id'] == self.vids_to_images[seq_id][t]['id']]
                for ann in timestep_annotations:
                    self.seq_properties[seq_id]['pos_category_ids'].add(ann['category_id'])
                    # convert box format from xywh to x0y0x1y1
                    lines.append('%d %d %d %d %d %d %d %d %d %s %f %f %f %f\n'
                                 % (t, ann['id'], ann['category_id'], ann['iscrowd'], 0, 0, 0, 0, 0, 'None',
                                    ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                                    ann['bbox'][1] + ann['bbox'][3]))
            data[seq] = lines
        return data

    def _write_clsmap_to_file(self):
        """
        Writes the class information to a file which will be located as split_to_convert.clsmap inside the new_gt_folder
        directory.
        The class information has the following fields:
            class_name(string), class_id(int), to_be_merged_into_class_id(int)
        The fields are separated by whitespaces.
        :return: None
        """
        lines = ['%s %d %d\n' % (k, v, self.merge_map[v]) for k, v in self.class_name_to_class_id.items()]
        clsmap_file = os.path.join(self.new_gt_folder, self.split_to_convert + '.clsmap')

        with open(clsmap_file, 'w') as f:
            f.writelines(lines)

    def _write_seqmap_to_file(self):
        """
        Writes the sequence meta information to a file which will be located as split_to_convert.seqmap inside the
        new_gt_folder directory.
        The sequence meta information has the following fields:
            sequence_name(string), sequence_length(int), positive_category_ids(separated by commas),
            negative_category_ids(separated by commas), not_exhaustively_labeled_category_ids(separated by commas)
        The fields are separated by whitespaces.
        :return: None
        """
        for vid in self.seq_properties.keys():
            self.seq_properties[vid]['pos_category_ids'] = list(self.seq_properties[vid]['pos_category_ids'])
        lines = ['%s %d %s %s %s\n' % (k, self.seq_properties[v]['length'],
                                       ','.join(str(i) for i in self.seq_properties[v]['pos_category_ids']),
                                       ','.join(str(i) for i in self.seq_properties[v]['neg_category_ids']),
                                       ','.join(str(i) for i in
                                                self.seq_properties[v]['not_exhaustively_labeled_ids']))
                 for k, v in self.sequences.items()]
        seqmap_file = os.path.join(self.new_gt_folder, self.split_to_convert + '.seqmap')

        with open(seqmap_file, 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    default_conf = TAOConverter.get_default_config()
    conf = utils.update_config(default_conf)
    TAOConverter(conf).convert()
