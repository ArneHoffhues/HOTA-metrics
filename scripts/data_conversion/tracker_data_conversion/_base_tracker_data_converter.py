import os
from pathlib import Path
import shutil
from abc import ABC, abstractmethod


class _BaseTrackerDataConverter(ABC):
    """
    Abstract base class for converting ground truth data into a unified format
    """
    @abstractmethod
    def __init__(self):
        self.tracker_fol = None
        self.new_tracker_folder = None
        self.tracker_list = None
        self.gt_dir = None
        self.output_as_zip = False
        self.split_to_convert = None
        self.config = None

    # Functions to implement:

    @staticmethod
    @abstractmethod
    def get_default_config():
        ...

    @staticmethod
    @abstractmethod
    def get_dataset_name():
        ...

    @abstractmethod
    def _prepare_data(self, tracker):
        ...

    # Helper function for all converters

    def convert(self):
        """
        Function that converts the ground truth data of a dataset into a unified format and writes the data into 3
        different files:
        1)  The actual data for each sequence is written to a text file which will be located inside the
            new_gt_folder/split_to_convert/ directory or will be compressed into a zip directory which will be located
            in the new_gt_folder directory.
        2)  The file containing the sequence meta information (seqmap) is written to a text file which will be located
            inside the new_gt_folder directory.
        3)  The file containing the class information (clsmap) is written to a text file which will be located inside
            the new_gt_folder directory.
        The unified format contains the following fields in this order:
            timestep(int), id(int), class(int), is_crowd(int), is_truncated(int), is_occluded(int), is_zero_marked(int),
            height (int), width(int), mask_counts(pycocotools-rle), bbox_x0(float), bbox_y0(float), bbox_x1(float),
            bbox_y1(float)
        The fields are separated by whitespaces.
        :return: None
        """
        print("Converting %s tracker data..." % self.get_dataset_name())
        for k, v in self.config.items():
            print("%s: %s" % (k, v))

        for tracker in self.tracker_list:
            print("Converting tracker data for %s..." % tracker)
            data = self._prepare_data(tracker)
            if not self.output_as_zip:
                data_dir = os.path.join(self.new_tracker_folder, tracker, 'data')
            else:
                data_dir = os.path.join(self.new_tracker_folder, tracker, 'tmp')
            # create directory if it does not exist
            Path(data_dir).mkdir(parents=True, exist_ok=True)

            for seq, lines in data.items():
                seq_file = os.path.join(data_dir, seq + '.txt')

                with open(seq_file, 'w') as f:
                    f.writelines(lines)

            # zip the output files and delete temporary data directory
            if self.output_as_zip:
                output_filename = os.path.join(self.new_tracker_folder, tracker, 'data')
                shutil.make_archive(output_filename, 'zip', data_dir)
                shutil.rmtree(data_dir)
