import os
import csv
from pathlib import Path
import shutil
from abc import ABC, abstractmethod


class _BaseTrackerDataConverter(ABC):
    """
    Abstract base class for converting tracker data into a unified format
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
        Function that converts the tracker data of a dataset into a unified format and writes the data into a file.
        The actual data for each sequence is written to a text file which will be located inside the
        new_tracker_folder/split_to_convert/ directory or will be compressed into a zip directory which will be located
        in the new_tracker_folder directory.

        The unified format contains the following fields in this order:
            timestep(int), id(int), class(int), height (int), width(int), mask_counts(pycocotools-rle), bbox_x0(float),
            bbox_y0(float), bbox_x1(float), bbox_y1(float), confidence_score(float)
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

    def _get_sequences(self):
        """
        Auxiliary function which reads the seqmap according to the split_to_convert which is located inside the
        gt_directory/seqmaps folder and extracts the sequence names, lengths and sizes.
        :return: None
        """
        if self.gt_dir:
            seqmap_file = os.path.join(self.gt_dir, 'seqmaps', self.split_to_convert + '.seqmap')
            if not os.path.isfile(seqmap_file):
                raise Exception('no seqmap found: ' + seqmap_file)
            self.seq_list = []
            self.seq_lengths = {}
            self.seq_sizes = {}
            with open(seqmap_file) as fp:
                dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    if len(row) >= 4:
                        seq = row[0]
                        self.seq_list.append(seq)
                        self.seq_lengths[seq] = int(row[1])
                        self.seq_sizes[seq] = (int(row[2]), int(row[3]))

    def _get_classes(self):
        """
        Auxiliary function which reads the slsmap according to the split_to_convert which is located inside the
        gt_directory/clsmaps folder and extracts the class names and corresponding IDs.
        :return: None
        """
        if self.gt_dir:
            clsmap_file = os.path.join(self.gt_dir, 'clsmaps', self.split_to_convert + '.clsmap')
            if not os.path.isfile(clsmap_file):
                raise Exception('No clsmap found: ' + clsmap_file)
            self.class_name_to_class_id = {}
            with open(clsmap_file) as fp:
                dialect = csv.Sniffer().sniff(fp.readline())
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    if len(row) == 2:
                        self.class_name_to_class_id[row[0]] = int(row[1])
                    # class names have white spaces
                    elif len(row) >= 3:
                        cls = ' '.join([elem for elem in row[:-1]])
                        self.class_name_to_class_id[cls] = int(row[-1])
