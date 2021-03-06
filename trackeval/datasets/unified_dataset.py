
import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException
from .. import _timing


class Unified(_BaseDataset):

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/converted_trackers'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'BENCHMARK': None,  # valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15', 'MOTS', 'kitti_2d_box', 'kitti_mots',
                                # 'bdd100k_2d_box', 'davis_unsupervised', 'tao', 'youtube_vis'
            'CLASSES_TO_EVAL': None,  # if None, all valid classes
            'SPLIT_TO_EVAL': None,
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/DATA_LOC_FORMAT/OUTPUT_SUB_FOLDER
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/DATA_LOC_FORMAT/TRACKER_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/dataset_subfolder/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use SEQMAP_FOLDER/BENCHMARK_SPLIT_TO_EVAL)
            'CLSMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/dataset_subfolder/clsmaps)
            'CLSMAP_FILE': None,  # Directly specify seqmap file (if none use CLSMAP_FOLDER/BENCHMARK_SPLIT_TO_EVAL)
            'DATA_LOC_FORMAT': '{dataset}/{benchmark}_{split}/',    # data localization format for GT, Tracker
                                                                    # and output subfolder structure
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        # associated dataset folder for benchmark
        self.benchmark = self.config['BENCHMARK']
        if self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20', 'MOTS']:
            self.dataset = 'mot_challenge'
        elif self.benchmark in ['kitti_2d_box', 'kitti_mots']:
            self.dataset = 'kitti'
        elif self.benchmark == 'davis_unsupervised':
            self.dataset = 'davis'
        elif self.benchmark == 'bdd100k_2d_box':
            self.dataset = 'bdd100k'
        elif self.benchmark == 'tao':
            self.dataset = 'tao'
        elif self.benchmark == 'youtube_vis':
            self.dataset = 'youtube_vis'
        else:
            raise TrackEvalException('Unknown Benchmark!')

        self.split = self.config['SPLIT_TO_EVAL']

        self.gt_fol = os.path.join(self.config['GT_FOLDER'])
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], self.config['DATA_LOC_FORMAT'].
                                        format(dataset=self.dataset, benchmark=self.benchmark, split=self.split))
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        else:
            self.output_fol = os.path.join(self.output_fol, self.config['DATA_LOC_FORMAT'].
                                           format(dataset=self.dataset, benchmark=self.benchmark, split=self.split))

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # kitti 2d box preprocessing parameters
        if self.benchmark == 'kitti_2d_box':
            self.max_occlusion = 2
            self.max_truncation = 0
            self.min_height = 25

        # read class and sequence maps
        self._get_cls_info()
        self._get_seq_info()

        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # get classes
        if self.benchmark == 'tao':
            # for TAO all classes present in ground truth data are valid
            pos_cat_id_list = list(set([cat for seq, cats in self.pos_categories.items() if seq
                                        in self.seq_list for cat in cats]))
            valid_classes = [cls for cls in self.class_name_to_class_id.keys() if self.class_name_to_class_id[cls]
                             in pos_cat_id_list]
        elif self.benchmark in ['kitti_2d_box', 'kitti_mots']:
            valid_classes = ['car', 'pedestrian']
        elif self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20', 'MOTS']:
            valid_classes = ['pedestrian']
        elif self.benchmark == 'davis_unsupervised':
            valid_classes = ['general']
        elif self.benchmark == 'bdd100k_2d_box':
            valid_classes = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']
        else:
            valid_classes = self.class_name_to_class_id.keys()

        if not self.config['CLASSES_TO_EVAL']:
            self.class_list = valid_classes
        else:
            self.class_list = [cls.lower() if cls.lower() in valid_classes else None
                               for cls in self.config['CLASSES_TO_EVAL']]
            if not all(self.class_list):
                raise TrackEvalException('Attempted to evaluate an invalid class. Only classes ' +
                                         ', '.join(valid_classes) + ' are valid.')

        self.should_classes_combine = True if self.benchmark in ['tao', 'bdd100k_2d_box', 'youtube_vis'] else False
        # set super categories for bdd100k evaluation
        if self.benchmark == 'bdd100k_2d_box':
            self.use_super_categories = True
            self.super_categories = {"HUMAN": [cls for cls in ["pedestrian", "rider"] if cls in self.class_list],
                                     "VEHICLE": [cls for cls in ["car", "truck", "bus", "train"]
                                                 if cls in self.class_list],
                                     "BIKE": [cls for cls in ["motorcycle", "bicycle"] if cls in self.class_list]}
        else:
            self.use_super_categories = False

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = os.path.join(self.gt_fol, self.config['DATA_LOC_FORMAT'].
                                         format(dataset=self.dataset, benchmark=self.benchmark, split=self.split),
                                         'data', seq + '.txt')
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, self.config['DATA_LOC_FORMAT'].
                                     format(dataset=self.dataset, benchmark=self.benchmark, split=self.split),
                                     'data.zip')
            if not os.path.isfile(curr_file):
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, 'data.zip')
                if not os.path.isfile(curr_file):
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol +
                            '/' + os.path.basename(curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        self.seq_list = []
        self.seq_lengths = {}
        self.seq_sizes = {}
        if self.benchmark == 'tao':
            self.pos_categories = {}
            self.neg_categories = {}
            self.not_exhaustively_labeled = {}
        if self.config["SEQMAP_FILE"]:
            seqmap_file = self.config["SEQMAP_FILE"]
        else:
            if self.config["SEQMAP_FOLDER"] is None:
                seqmap_file = os.path.join(self.config['GT_FOLDER'], self.dataset, 'seqmaps',
                                           self.benchmark + '_' + self.split + '.seqmap')
            else:
                seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.benchmark + '_' + self.split + '.seqmap')
        if not os.path.isfile(seqmap_file):
            raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
        with open(seqmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for i, row in enumerate(reader):
                if len(row) >= 4:
                    # first col: sequence, second col: seqeuence length, third and fourth col: sequence height/width
                    seq = row[0]
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[1])
                    self.seq_sizes[seq] = (int(row[2]), int(row[3]))
                if len(row) >= 7:
                    # for TAO positive categories (classes present in ground truth data) are listed in column 5,
                    # negative categories (potential false negatives during evaluation) are listed in column 6,
                    # not exhaustively labeled categories are listed in column 7
                    self.pos_categories[seq] = [int(cat) for cat in row[4].split(',')]
                    self.neg_categories[seq] = [int(cat) for cat in row[5].split(',')] if len(row[5]) > 0 else []
                    self.not_exhaustively_labeled[seq] = [int(cat) for cat in row[6].split(',')] \
                        if len(row[6]) > 0 else []

    def _get_cls_info(self):
        self.class_name_to_class_id = {}
        if self.benchmark == 'tao':
            self.merge_map = {}
        if self.config["CLSMAP_FILE"]:
            clsmap_file = self.config["CLSMAP_FILE"]
        else:
            if self.config["CLSMAP_FOLDER"] is None:
                clsmap_file = os.path.join(self.config['GT_FOLDER'], self.dataset, 'clsmaps',
                                           self.benchmark + '_' + self.split + '.clsmap')
            else:
                clsmap_file = os.path.join(self.config["CLSMAP_FOLDER"], self.benchmark + '_' + self.split + '.clsmap')
        if not os.path.isfile(clsmap_file):
            raise TrackEvalException('no clsmap found: ' + os.path.basename(clsmap_file))
        with open(clsmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for i, row in enumerate(reader):
                if len(row) == 2:
                    # column 1: class names, column 2: class IDs
                    cls = row[0]
                    self.class_name_to_class_id[cls] = int(row[1])
                else:
                    # row length > 2: class names contain either white spaces or the benchmark is TAO as TAO contains
                    # an additional field for categories that are to be merged
                    if self.benchmark == 'tao':
                        cls = ' '.join([entry for entry in row[:-2]])
                        self.class_name_to_class_id[cls] = int(row[-2])
                        if row[-2] != row[-1]:
                            self.merge_map[int(row[-2])] = int(row[-1])
                    else:
                        cls = ' '.join([entry for entry in row[:-1]])
                        self.class_name_to_class_id[cls] = int(row[-1])

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the Unified format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).
        Extra fields for TAO and YouTubeVIS benchmark:
        [classes_to_gt_tracks]: nested dictionary with class values and track IDs as keys and list of dictionaries
                                (with frame indices as keys and corresponding detection as values) for each track
                                as values
        [classes_to_gt_track_iscrowd]:  nested dictionary with class values and track IDs as keys and
                                        lists (for each track) as values

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        Extra fields for TAO and YouTubeVIS benchmark:
        [classes_to_dt_tracks]: nested dictionary with class values and track IDs as keys and list of dictionaries
                                (with frame indices as keys and corresponding detection as values) for each track
                                as values
        [classes_to_track_scores]:  nested dictionary with class values and track IDs as keys and
                                    lists (for each track) as values
        """
        # import to reduce minimum requirements
        from pycocotools import mask as mask_utils

        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, self.config['DATA_LOC_FORMAT'].
                                        format(dataset=self.dataset, benchmark=self.benchmark, split=self.split),
                                        'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, 'data.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = os.path.join(self.gt_fol, self.config['DATA_LOC_FORMAT'].
                                    format(dataset=self.dataset, benchmark=self.benchmark, split=self.split),
                                    'data', seq + '.txt')
            else:
                file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')

        # set crowd ignore filter
        if is_gt:
            if self.benchmark in ['MOTS', 'kitti_mots']:
                crowd_ignore_filter = {2: [str(self.class_name_to_class_id['ignore'])]}
            elif self.benchmark == 'kitti_2d_box':
                crowd_ignore_filter = {2: [str(self.class_name_to_class_id['dontcare'])]}
            elif self.benchmark == 'bdd100k_2d_box':
                distractor_class_names = ['other person', 'trailer', 'other vehicle']
                crowd_ignore_filter = {10: ['1'], 2: [str(self.class_name_to_class_id[x])
                                                      for x in distractor_class_names]}
            # for davis void pixels are read as crowd ignore
            elif self.benchmark == 'davis_unsupervised':
                crowd_ignore_filter = {2: [str(self.class_name_to_class_id['void'])]}
            else:
                crowd_ignore_filter = None
        else:
            crowd_ignore_filter = None

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, crowd_ignore_filter=crowd_ignore_filter,
                                                             is_zipped=self.data_is_zipped, zip_file=zip_file,
                                                             force_delimiters=' ')

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for t in range(num_timesteps):
            time_key = str(t)
            # list to collect all masks of a timestep to check for overlapping areas (for segmentation datasets)
            all_masks = []
            if time_key in read_data.keys():
                try:
                    raw_data['ids'][t] = np.atleast_1d([det[1] for det in read_data[time_key]]).astype(int)
                    raw_data['classes'][t] = np.atleast_1d([det[2] for det in read_data[time_key]]).astype(int)
                    # merge categories for TAO
                    if self.benchmark == 'tao':
                        raw_data['classes'][t] = np.atleast_1d([self.merge_map.get(cls, cls) for cls
                                                                in raw_data['classes'][t]]).astype(int)
                    if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
                        raw_data['dets'][t] = [{'size': [int(region[3]), int(region[4])],
                                                'counts': region[5].encode(encoding='UTF-8')}
                                               for region in read_data[time_key]]
                        all_masks += raw_data['dets'][t]
                    else:
                        raw_data['dets'][t] = np.atleast_2d([det[6:10] for det in read_data[time_key]]).astype(float)
                    if is_gt:
                        gt_extras_dict = {'crowd': np.atleast_1d([det[10] for det in read_data[time_key]]).astype(int),
                                          'truncation': np.atleast_1d([det[11] for det
                                                                       in read_data[time_key]]).astype(int),
                                          'occlusion': np.atleast_1d([det[12] for det
                                                                      in read_data[time_key]]).astype(int),
                                          'zero_marked': np.atleast_1d([det[13] for det
                                                                        in read_data[time_key]]).astype(int)}
                        raw_data['gt_extras'][t] = gt_extras_dict
                    else:
                        raw_data['tracker_confidences'][t] = np.atleast_1d([det[10] for det
                                                                            in read_data[time_key]]).astype(float)
                except IndexError:
                    self._raise_index_error(is_gt, tracker, seq)
                except ValueError:
                    self._raise_value_error(is_gt, tracker, seq)
            # no detection in this timestep
            else:
                if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
                    raw_data['dets'][t] = []
                else:
                    raw_data['dets'][t] = np.empty((0, 4)).astype(float)
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'crowd': np.empty(0).astype(int),
                                      'truncation': np.empty(0).astype(int),
                                      'occlusion': np.empty(0).astype(int),
                                      'zero_marked': np.empty(0).astype(int)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0).astype(float)

            if is_gt:
                if time_key in ignore_data.keys():
                    try:
                        if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
                            time_ignore = [{'size': [int(region[3]), int(region[4])],
                                            'counts': region[5].encode(encoding='UTF-8')}
                                           for region in ignore_data[time_key]]
                            raw_data['gt_ignore_regions'][t] = mask_utils.merge([mask for mask in time_ignore],
                                                                                intersect=False)
                            # for davis ignore data (void pixels) may overlap with other masks
                            if not self.benchmark == 'davis_unsupervised':
                                all_masks += time_ignore
                        else:
                            raw_data['gt_ignore_regions'][t] = np.atleast_2d([det[6:10] for det
                                                                              in ignore_data[time_key]]).astype(float)
                    except IndexError:
                        self._raise_index_error(is_gt, tracker, seq)
                    except ValueError:
                        self._raise_value_error(is_gt, tracker, seq)
                # no ignore data for this timestep
                else:
                    if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
                        raw_data['gt_ignore_regions'][t] = mask_utils.merge([], intersect=False)
                    else:
                        raw_data['gt_ignore_regions'][t] = np.empty((0, 4)).astype(float)

            # check for overlapping masks
            if all_masks:
                masks_merged = all_masks[0]
                for mask in all_masks[1:]:
                    if mask_utils.area(mask_utils.merge([masks_merged, mask], intersect=True)) != 0.0:
                        err = 'Overlapping masks in frame %d' % t
                        raise TrackEvalException(err)
                    masks_merged = mask_utils.merge([masks_merged, mask], intersect=False)

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}

        # assemble per frame detections to tracks for TAO and YouTubeVIS in order to evaluate TrackMAP
        if self.benchmark in ['tao', 'youtube_vis']:
            raw_data['classes_to_tracks'] = {}
            if not is_gt:
                raw_data['classes_to_track_scores'] = {}

            # classes to consider
            if self.benchmark == 'youtube_vis' or self.benchmark == 'tao' and is_gt:
                classes_to_consider = [self.class_name_to_class_id[cls] for cls in self.class_list]
            elif self.benchmark == 'tao' and not is_gt:
                classes_to_consider = self.pos_categories[seq] + self.neg_categories[seq]
            else:
                raise TrackEvalException('Track based evaluation undefined for benchmark %s' % self.benchmark)

            # assemble tracks and get corresponding track scores
            for t in range(num_timesteps):
                for i in range(len(raw_data['ids'][t])):
                    tid = raw_data['ids'][t][i]
                    cls_id = raw_data['classes'][t][i]
                    if cls_id in classes_to_consider:
                        if cls_id not in raw_data['classes_to_tracks']:
                            raw_data['classes_to_tracks'][cls_id] = {}
                        if tid not in raw_data['classes_to_tracks'][cls_id]:
                            raw_data['classes_to_tracks'][cls_id][tid] = {}
                        raw_data['classes_to_tracks'][cls_id][tid][t] = raw_data['dets'][t][i]
                        if not is_gt:
                            if cls_id not in raw_data['classes_to_track_scores']:
                                raw_data['classes_to_track_scores'][cls_id] = {}
                            if tid not in raw_data['classes_to_track_scores'][cls_id]:
                                raw_data['classes_to_track_scores'][cls_id][tid] = []
                            raw_data['classes_to_track_scores'][cls_id][tid].\
                                append(raw_data['tracker_confidences'][t][i])

            # fill missing classes with default values
            for cls in self.class_list:
                cls_id = self.class_name_to_class_id[cls]
                if cls_id not in raw_data['classes_to_tracks']:
                    raw_data['classes_to_tracks'][cls_id] = {}
                if not is_gt and cls_id not in raw_data['classes_to_track_scores']:
                    raw_data['classes_to_track_scores'][cls_id] = []

            # get crowd ignore data for tracks
            if self.benchmark == 'youtube_vis' and is_gt:
                raw_data['classes_to_gt_track_iscrowd'] = {}
                for t in range(num_timesteps):
                    for i in range(len(raw_data['ids'][t])):
                        tid = raw_data['ids'][t][i]
                        cls_id = raw_data['classes'][t][i]
                        if cls_id in classes_to_consider:
                            if cls_id not in raw_data['classes_to_gt_track_iscrowd']:
                                raw_data['classes_to_gt_track_iscrowd'][cls_id] = {}
                            if tid not in raw_data['classes_to_gt_track_iscrowd'][cls_id]:
                                raw_data['classes_to_gt_track_iscrowd'][cls_id][tid] = []
                            raw_data['classes_to_gt_track_iscrowd'][cls_id][tid].\
                                append(raw_data['gt_extras'][t]['crowd'][i])

                # fill missing classes with default values
                for cls in self.class_list:
                    cls_id = self.class_name_to_class_id[cls]
                    if cls_id not in raw_data['classes_to_gt_track_iscrowd']:
                        raw_data['classes_to_gt_track_iscrowd'][cls_id] = {}
                    else:
                        raw_data['classes_to_gt_track_iscrowd'][cls_id] = \
                            {k: np.all(v).astype(int) for k, v in
                             raw_data['classes_to_gt_track_iscrowd'][cls_id].items()}

            if is_gt:
                key_map['classes_to_tracks'] = 'classes_to_gt_tracks'
            else:
                key_map['classes_to_tracks'] = 'classes_to_dt_tracks'

        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        # parameters for TAO preprocessing
        if self.benchmark == 'tao':
            raw_data['neg_cat_ids'] = self.neg_categories[seq]
            raw_data['not_exhaustively_labeled_cls'] = self.not_exhaustively_labeled[seq]
        raw_data['num_timesteps'] = num_timesteps
        raw_data['frame_size'] = self.seq_sizes[seq]
        raw_data['seq'] = seq
        return raw_data

    @staticmethod
    def _raise_index_error(is_gt, tracker, seq):
        """
        Auxiliary method to raise an evaluation error in case of an index error while reading files.
        :param is_gt: whether gt or tracker data is read
        :param tracker: the name of the tracker
        :param seq: the name of the seq
        :return: None
        """
        if is_gt:
            err = 'Cannot load gt data from sequence %s, because there are not enough ' \
                  'columns in the data.' % seq
            raise TrackEvalException(err)
        else:
            err = 'Cannot load tracker data from tracker %s, sequence %s, because there are not enough ' \
                  'columns in the data.' % (tracker, seq)
            raise TrackEvalException(err)

    @staticmethod
    def _raise_value_error(is_gt, tracker, seq):
        """
        Auxiliary method to raise an evaluation error in case of an value error while reading files.
        :param is_gt: whether gt or tracker data is read
        :param tracker: the name of the tracker
        :param seq: the name of the seq
        :return: None
        """
        if is_gt:
            raise TrackEvalException(
                'GT data for sequence %s cannot be converted to the right format. Is data corrupted?' % seq)
        else:
            raise TrackEvalException(
                'Tracking data from tracker %s, sequence %s cannot be converted to the right format. '
                'Is data corrupted?' % (tracker, seq))

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.

        KITTI:
            In KITTI, the 4 preproc steps are as follow:
                1) There are two classes (pedestrian and car) which are evaluated separately.
                2) For the pedestrian class, the 'person' class is distractor objects (people sitting).
                    For the car class, the 'van' class are distractor objects.
                    GT boxes marked as having occlusion level > 2 or truncation level > 0 are also treated as
                        distractors.
                3) Crowd ignore regions are used to remove unmatched detections. Also unmatched detections with
                    height <= 25 pixels are removed.
                4) Distractor gt dets (including truncated and occluded) are removed.

        MOTS and KITTI MOTS:
            In MOTS, the 4 preproc steps are as follow:
                1) There are two classes (car and pedestrian) which are evaluated separately.
                2) There are no ground truth detections marked as to be removed/distractor classes.
                    Therefore also no matched tracker detections are removed.
                3) Ignore regions are used to remove unmatched detections (at least 50% overlap with ignore region).
                4) There are no ground truth detections (e.g. those of distractor classes) to be removed.

        TAO:
            In TAO, the 4 preproc steps are as follow:
                1) All classes present in the ground truth data are evaluated separately.
                2) No matched tracker detections are removed.
                3) Unmatched tracker detections are removed if there is not ground truth data and the class does not
                    belong to the categories marked as negative for this sequence. Additionally, unmatched tracker
                    detections for classes which are marked as not exhaustively labeled are removed.
                4) No gt detections are removed.
            Further, for TrackMAP computation track representations for the given class are accessed from a dictionary
            and the tracks from the tracker data are sorted according to the tracker confidence.

        YouTubeVIS:
            In YouTubeVIS, the 4 preproc steps are as follow:
                1) There are 40 classes which are evaluated separately.
                2) No matched tracker dets are removed.
                3) No unmatched tracker dets are removed.
                4) No gt dets are removed.
            Further, for TrackMAP computation track representations for the given class are accessed from a dictionary
            and the tracks from the tracker data are sorted according to the tracker confidence.

        BDD100K:
            In BDD100K, the 4 preproc steps are as follow:
                1) There are eight classes (pedestrian, rider, car, bus, truck, train, motorcycle, bicycle)
                    which are evaluated separately.
                2) For BDD100K there is no removal of matched tracker dets.
                3) Crowd ignore regions are used to remove unmatched detections.
                4) No removal of gt dets.

        DAVIS:
            In DAVIS, the 4 preproc steps are as follow:
                1) There are no classes, all detections are evaluated jointly
                2) No matched tracker detections are removed.
                3) No unmatched tracker detections are removed.
                4) There are no ground truth detections (e.g. those of distractor classes) to be removed.
            Preprocessing special to DAVIS: Pixels which are marked as void in the ground truth are set to zero in the
                tracker detections since they are not considered during evaluation.
        """
        # import to reduce minimum requirements
        from pycocotools import mask as mask_utils

        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        # get distractor classes
        if self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20']:
            distractor_class_names = ['person_on_vehicle', 'static_person', 'distractor', 'reflection']
            if self.benchmark == 'MOT20':
                distractor_class_names.append('non_mot_vehicle')
        elif self.benchmark == 'kitti_2d_box':
            if cls == 'pedestrian':
                distractor_class_names = ['person']
            elif cls == 'car':
                distractor_class_names = ['van']
            else:
                raise (TrackEvalException('Class %s is not evaluatable' % cls))
        else:
            distractor_class_names = []
        distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        cls_id = self.class_name_to_class_id[cls]

        # get preprocessing parameters for TAO
        is_neg_category = cls_id in raw_data['neg_cat_ids'] if self.benchmark == 'tao' else False
        is_not_exhaustively_labeled = cls_id in raw_data['not_exhaustively_labeled_cls'] if self.benchmark == 'tao' \
            else False

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):

            if self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20']:
                # in MOT all ground truth detections are considered
                gt_class_mask = np.ones(len(raw_data['gt_classes'][t])).astype(np.bool)
            else:
                # Only extract relevant dets for this class for preproc and eval (cls + distractor classes)
                gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in [cls_id] + distractor_classes], axis=0)
                gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
                gt_dets = [raw_data['gt_dets'][t][ind] for ind in range(len(gt_class_mask)) if gt_class_mask[ind]]
            else:
                gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]
            gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked'][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
                tracker_dets = [raw_data['tracker_dets'][t][ind] for ind in range(len(tracker_class_mask)) if
                                tracker_class_mask[ind]]
            else:
                tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            if self.benchmark == 'youtube_vis':
                # no preprocessing for YouTubeVIS
                data['tracker_ids'][t] = tracker_ids
                data['tracker_dets'][t] = tracker_dets
                data['gt_ids'][t] = gt_ids
                data['gt_dets'][t] = gt_dets
                data['similarity_scores'][t] = similarity_scores
            elif self.benchmark == 'davis_unsupervised':
                # for DAVIS only set void pixels to zero
                data['tracker_ids'][t] = tracker_ids
                data['gt_ids'][t] = gt_ids
                data['gt_dets'][t] = gt_dets
                data['similarity_scores'][t] = similarity_scores

                # set void pixels in tracker detections to zero
                void_mask = raw_data['gt_ignore_regions'][t]
                if mask_utils.area(void_mask) > 0:
                    void_mask_ious = np.\
                        atleast_1d(mask_utils.iou(tracker_dets, [void_mask],
                                                  [False for _ in range(len(tracker_dets))]))
                    if void_mask_ious.any():
                        rows, columns = np.where(void_mask_ious > 0)
                        for r in rows:
                            det = mask_utils.decode(tracker_dets[r])
                            void = mask_utils.decode(void_mask).astype(np.bool)
                            det[void] = 0
                            det = mask_utils.encode(np.array(det, order='F').astype(np.uint8))
                            tracker_dets[r] = det
                data['tracker_dets'][t] = tracker_dets
            else:
                # Match tracker and gt dets (with hungarian algorithm)
                to_remove_matched = np.array([], np.int)
                unmatched_indices = np.arange(tracker_ids.shape[0])
                if self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                    matching_scores = similarity_scores.copy()
                    matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                    match_rows, match_cols = linear_sum_assignment(-matching_scores)
                    actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                    match_rows = match_rows[actually_matched_mask]
                    match_cols = match_cols[actually_matched_mask]

                    if self.benchmark == 'kitti_2d_box':
                        # remove tracker dets which match with gt dets which are labeled as truncated, occluded,
                        # or belonging to a distractor class.
                        is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                        is_occluded_or_truncated = np.logical_or(gt_occlusion[match_rows] > self.max_occlusion,
                                                                 gt_truncation[match_rows] > self.max_truncation)
                        to_remove_matched = np.logical_or(is_distractor_class, is_occluded_or_truncated)
                        to_remove_matched = match_cols[to_remove_matched]
                    elif self.benchmark in ['MOT16', 'MOT17', 'MOT20']:
                        # remove tracker dets which match with gt dets belonging to a distractor class.
                        is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                        to_remove_matched = match_cols[is_distractor_class]
                    unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

                if self.benchmark in ['kitti_2d_box', 'bdd100k_2d_box']:
                    # For unmatched tracker dets remove those that are greater than 50% within a crowd ignore region.
                    unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
                    crowd_ignore_regions = raw_data['gt_ignore_regions'][t]
                    intersection_with_ignore_region = self.\
                        _calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions, box_format='x0y0x1y1',
                                            do_ioa=True)
                    is_within_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps,
                                                     axis=1)
                    if self.benchmark == 'kitti_2d_box':
                        # For unmatched tracker dets, also remove those smaller than a minimum height.
                        unmatched_heights = unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
                        is_too_small = unmatched_heights <= self.min_height

                        to_remove_unmatched = unmatched_indices[np.logical_or(is_too_small,
                                                                              is_within_ignore_region)]
                        to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched), axis=0)
                    else:
                        to_remove_tracker = unmatched_indices[is_within_ignore_region]
                elif self.benchmark in ['kitti_mots', 'MOTS']:
                    # For unmatched tracker dets remove those that are greater than 50% within a crowd ignore region.
                    unmatched_tracker_dets = [tracker_dets[i] for i in range(len(tracker_dets))
                                              if i in unmatched_indices]
                    ignore_region = raw_data['gt_ignore_regions'][t]
                    intersection_with_ignore_region = self.\
                        _calculate_mask_ious(unmatched_tracker_dets, [ignore_region], is_encoded=True, do_ioa=True)
                    is_within_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps,
                                                     axis=1)
                    to_remove_tracker = unmatched_indices[is_within_ignore_region]
                elif self.benchmark == 'tao':
                    # for TAO remove unmatched tracker detections if there is no ground truth data and the category
                    # is not marked as a negative category or the category is not exhaustively labeled
                    if gt_ids.shape[0] == 0 and not is_neg_category:
                        to_remove_tracker = unmatched_indices
                    elif is_not_exhaustively_labeled:
                        to_remove_tracker = unmatched_indices
                    else:
                        to_remove_tracker = np.array([], dtype=np.int)
                elif self.benchmark in ['MOT16', 'MOT17', 'MOT20']:
                    to_remove_tracker = to_remove_matched
                else:
                    to_remove_tracker = np.array([], dtype=np.int)

                # remove all unwanted tracker detections
                data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
                data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
                data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
                similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

                if self.benchmark in ['kitti_2d_box', 'MOT15', 'MOT16', 'MOT17', 'MOT20']:
                    if self.benchmark == 'kitti_2d_box':
                        # Remove gt dets that were only useful for preprocessing and are not needed for evaluation.
                        # These are those that are occluded, truncated and from distractor objects.
                        gt_to_keep_mask = (np.less_equal(gt_occlusion, self.max_occlusion)) & \
                                          (np.less_equal(gt_truncation, self.max_truncation)) & \
                                          (np.equal(gt_classes, cls_id))
                    elif self.benchmark in ['MOT16', 'MOT17', 'MOT20']:
                        # Remove gt detections marked as to remove (zero marked), and also remove gt detections
                        # not in pedestrian class
                        gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                          (np.equal(gt_classes, cls_id))
                    else:
                        # There are no classes for MOT15
                        gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
                    data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
                    data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
                    data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]
                else:
                    # for the rest of the benchmarks keep all ground truth detections
                    data['gt_ids'][t] = gt_ids
                    data['gt_dets'][t] = gt_dets
                    data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        data['frame_size'] = raw_data['frame_size']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        # get track based information for TrackMAP evaluation
        if self.benchmark in ['tao', 'youtube_vis']:
            data['gt_track_ids'] = [key for key in raw_data['classes_to_gt_tracks'][cls_id].keys()]
            data['gt_tracks'] = [raw_data['classes_to_gt_tracks'][cls_id][tid] for tid in data['gt_track_ids']]
            data['gt_track_lengths'] = [len(track.keys()) for track in data['gt_tracks']]
            data['dt_track_ids'] = [key for key in raw_data['classes_to_dt_tracks'][cls_id].keys()]
            data['dt_tracks'] = [raw_data['classes_to_dt_tracks'][cls_id][tid] for tid in data['dt_track_ids']]
            data['dt_track_lengths'] = [len(track.keys()) for track in data['dt_tracks']]
            data['dt_track_scores'] = [np.mean(raw_data['classes_to_track_scores'][cls_id][tid]) for tid
                                       in data['dt_track_ids']]

            if self.benchmark == 'tao':
                data['iou_type'] = 'bbox'
                data['boxformat'] = 'x0y0x1y1'
                data['not_exhaustively_labeled'] = is_not_exhaustively_labeled
                data['gt_track_areas'] = []
                for tid in data['gt_track_ids']:
                    track = raw_data['classes_to_gt_tracks'][cls_id][tid]
                    if track:
                        data['gt_track_areas'].append(sum([(ann[2] - ann[0]) * (ann[3] - ann[1]) for ann
                                                           in track.values()]) / len(track.keys()))
                    else:
                        data['gt_track_areas'].append(0)
                data['dt_track_areas'] = []
                for tid in data['dt_track_ids']:
                    track = raw_data['classes_to_dt_tracks'][cls_id][tid]
                    if track:
                        data['dt_track_areas'].append(sum([(ann[2] - ann[0]) * (ann[3] - ann[1]) for ann
                                                           in track.values()]) / len(track.keys()))
                    else:
                        data['dt_track_areas'].append(0)

            if self.benchmark == 'youtube_vis':
                data['iou_type'] = 'mask'
                data['gt_track_iscrowd'] = [raw_data['classes_to_gt_track_iscrowd'][cls_id][tid]
                                            for tid in data['gt_track_ids']]

                for key in ['gt', 'dt']:
                    data[key + '_track_areas'] = []
                    for tid in data[key + '_track_ids']:
                        track = raw_data['classes_to_' + key + '_tracks'][cls_id][tid]
                        if track:
                            areas = []
                            for seg in track.values():
                                if seg:
                                    areas.append(mask_utils.area(seg))
                                else:
                                    areas.append(None)
                            areas = [a for a in areas if a]
                            if len(areas) == 0:
                                data[key + '_track_areas'].append(0)
                            else:
                                data[key + '_track_areas'].append(np.array(areas).mean())

            # sort tracker tracks by tracker score
            if data['dt_tracks']:
                idx = np.argsort([-score for score in data['dt_track_scores']], kind="mergesort")
                data['dt_track_scores'] = [data['dt_track_scores'][i] for i in idx]
                data['dt_tracks'] = [data['dt_tracks'][i] for i in idx]
                data['dt_track_ids'] = [data['dt_track_ids'][i] for i in idx]
                data['dt_track_lengths'] = [data['dt_track_lengths'][i] for i in idx]
                data['dt_track_areas'] = [data['dt_track_areas'][i] for i in idx]

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if self.benchmark in ['davis_unsupervised', 'youtube_vis', 'MOTS', 'kitti_mots']:
            similarity_scores = self._calculate_mask_ious(gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False)
        else:
            similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores
