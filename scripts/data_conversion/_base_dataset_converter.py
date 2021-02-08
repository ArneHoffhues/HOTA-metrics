import os
from pathlib import Path
import shutil
from abc import ABC, abstractmethod


class _BaseDatasetConverter(ABC):
    """
    Abstract base class for converting ground truth data into a unified format
    """
    @abstractmethod
    def __init__(self):
        self.seq_list = None
        self.seq_lengths = None
        self.class_name_to_class_id = None
        self.gt_fol = None
        self.new_gt_folder = None
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
    def _prepare_data(self):
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
        print("Converting %s ground truth data..." % self.get_dataset_name())
        for k, v in self.config.items():
            print("%s: %s" % (k, v))

        data = self._prepare_data()
        self._write_data_to_file(data)
        self._write_seqmap_to_file()
        self._write_clsmap_to_file()

    def _write_data_to_file(self, data):
        """
        Writes the data for each sequence into a different file. The file will be located inside the
        new_gt_folder/split_to_convert/ directory or will be compressed into a zip directory which will be located
        in the new_gt_folder directory.
        :param data: a dictionary which maps the sequence names to the lines that should be written to the according
                     sequence file
        :return: None
        """
        if not self.output_as_zip:
            data_dir = os.path.join(self.new_gt_folder, self.split_to_convert, 'data')
        else:
            data_dir = os.path.join(self.new_gt_folder, self.split_to_convert, 'tmp')
        # create directory if it does not exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        for seq, lines in data.items():
            seq_file = os.path.join(data_dir, seq + '.txt')

            with open(seq_file, 'w') as f:
                f.writelines(lines)

        # zip the output files and delete temporary data directory
        if self.output_as_zip:
            output_filename = os.path.join(self.new_gt_folder, self.split_to_convert, 'data')
            shutil.make_archive(output_filename, 'zip', data_dir)
            shutil.rmtree(data_dir)

    def _write_clsmap_to_file(self):
        """
        Writes the class information to a file which will be located as split_to_convert.clsmap inside the new_gt_folder
        directory.
        The class information has the following fields:
            class_name(string), class_id(int)
        The fields are separated by whitespaces.
        :return: None
        """
        lines = ['%s %d\n' % (k, v) for k, v in self.class_name_to_class_id.items()]
        Path(os.path.join(self.new_gt_folder, 'clsmaps')).mkdir(parents=True, exist_ok=True)
        clsmap_file = os.path.join(self.new_gt_folder, 'clsmaps', self.split_to_convert + '.clsmap')

        with open(clsmap_file, 'w') as f:
            f.writelines(lines)

    def _write_seqmap_to_file(self):
        """
        Writes the sequence meta information to a file which will be located as split_to_convert.seqmap inside the
        new_gt_folder directory.
        The sequence meta information has the following fields:
            sequence_name(string), sequence_length(int)
        The fields are separated by whitespaces.
        :return: None
        """
        lines = ['%s %d\n' % (k, v) for k, v in self.seq_lengths.items()]
        Path(os.path.join(self.new_gt_folder, 'seqmaps')).mkdir(parents=True, exist_ok=True)
        seqmap_file = os.path.join(self.new_gt_folder, 'seqmaps', self.split_to_convert + '.seqmap')

        with open(seqmap_file, 'w') as f:
            f.writelines(lines)
