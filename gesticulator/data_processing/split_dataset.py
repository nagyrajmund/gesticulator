"""
This script is used to split the dataset into train, test and dev sets.
More info on its usage is given in the main README.md file 

@authors: Taras Kucherenko, Rajmund Nagy
"""
import sys
import os
import shutil
import pandas
from os import path

# Params
from gesticulator.data_processing.data_params import dataset_argparser

# Indices for train/dev/test split
DEV_LAST_ID = 3
TEST_LAST_ID = 4
TRAIN_LAST_ID = 24

audio_prefix = "Recording_"
motion_prefix = "Recording_"

def copy_files(ind, raw_d_dir, processed_d_dir, data_split, use_mirror_augment = False, suffix=""):

    # add leading zeros
    ind = str(ind).zfill(3)

    # Copy audio
    filename = f"{audio_prefix}{ind}{suffix}.wav"
    original_file_path = path.join(raw_d_dir, "Audio", filename)

    if os.path.isfile(original_file_path):
        target_file_path = path.join(processed_d_dir, data_split, "inputs", filename)
        shutil.copy(original_file_path, target_file_path)

    # Copy text
    filename = f"{audio_prefix}{ind}{suffix}.json"
    transcript_file_path = path.join(raw_d_dir, "Transcripts", filename)
    
    if os.path.isfile(transcript_file_path):
        target_file_path = path.join(processed_d_dir, data_split, "inputs", filename)
        shutil.copy(transcript_file_path, target_file_path)

    # Copy gestures
    filename = f"{motion_prefix}{ind}{suffix}.npz"
    original_file_path = path.join(raw_d_dir, "Motion", filename)
    
    if os.path.isfile(original_file_path):
        target_file_path = path.join(processed_d_dir, data_split, "labels", filename)
        shutil.copy(original_file_path, target_file_path)      

    # Also copy mirrored gestures if they're enabled
    if use_mirror_augment:
        assert(data_split == "train")
        # Put _mirrored before .npz extension
        filename = filename[:-4] + "_mirrored" + filename[-4:]
        original_file_path = path.join(raw_d_dir, "Motion_mirrored", filename)

        if os.path.isfile(original_file_path):
            target_file_path = path.join(processed_d_dir, data_split, "labels_mirrored", filename)
            shutil.copy(original_file_path, target_file_path)      


def create_dataset_splits(raw_d_dir, processed_d_dir, use_mirror_augment):
    """Create the train/dev/test splits in new subfolders within 'processed_d_dir'."""
    _create_data_directories(processed_d_dir, use_mirror_augment)

    # prepare dev data
    for i in range(1, DEV_LAST_ID):
        copy_files(i, raw_d_dir, processed_d_dir, "dev")

    # prepare test data
    for i in range(DEV_LAST_ID, TEST_LAST_ID):
        copy_files(i, raw_d_dir, processed_d_dir, "test")

    # prepare training data
    for i in range(TEST_LAST_ID, TRAIN_LAST_ID):
        copy_files(i, raw_d_dir, processed_d_dir, "train", use_mirror_augment)

    extracted_dir = path.join(processed_d_dir)

    if use_mirror_augment:
        dev_files, train_files, test_files, mirrored_train_files = _format_datasets(extracted_dir, use_mirror_augment=True)
        mirrored_train_files.to_csv(path.join(extracted_dir, "train_mirrored-dataset-info.csv"), index=False)
    else:
        dev_files, train_files, test_files = _format_datasets(extracted_dir, use_mirror_augment)

    # Save the filenames of each datapoints (the preprocessing script will use these)
    dev_files.to_csv(path.join(extracted_dir, "dev-dataset-info.csv"), index=False)
    train_files.to_csv(path.join(extracted_dir, "train-dataset-info.csv"), index=False)
    test_files.to_csv(path.join(extracted_dir, "test-dataset-info.csv"), index=False)
        

def _create_data_directories(processed_d_dir, use_mirror_augment):
    """Create subdirectories for the dataset splits."""
    dir_names = ["dev", "test", "train"]
    sub_dir_names = ["inputs", "labels"]
    if use_mirror_augment:
        sub_dir_names.append("labels_mirrored")
    os.makedirs(processed_d_dir, exist_ok = True)
    
    print("Creating the datasets in the following directories:") 
    for dir_name in dir_names:
        dir_path = path.join(processed_d_dir, dir_name)
        print('  ', path.abspath(dir_path))
        os.makedirs(dir_path, exist_ok=True)  # e.g. ../../dataset/processed/train

        for sub_dir_name in sub_dir_names:
            dir_path = path.join(processed_d_dir, dir_name, sub_dir_name)
            os.makedirs(dir_path, exist_ok = True) # e.g. ../../dataset/processed/train/inputs/
    print()


def _format_datasets(extracted_dir, use_mirror_augment):
    print("The datasets will contain the following indices:", end='')
    dev_files = _files_to_pandas_dataframe(extracted_dir, "dev", range(1, DEV_LAST_ID))
    test_files = _files_to_pandas_dataframe(extracted_dir, "test", range(DEV_LAST_ID, TEST_LAST_ID))
    train_files = _files_to_pandas_dataframe(extracted_dir, "train", range(TEST_LAST_ID, TRAIN_LAST_ID))
    
    # Also copy the mirrored files, if enabled
    if use_mirror_augment:
        mirrored_files = _files_to_pandas_dataframe(extracted_dir, "train", range(TEST_LAST_ID, TRAIN_LAST_ID), use_mirror_augment=True)
        print()
        return dev_files, train_files, test_files, mirrored_files
    else:
        print()
        return dev_files, train_files, test_files
    
def _files_to_pandas_dataframe(extracted_dir, set_name, idx_range, use_mirror_augment=False):
    info_msg = f"\n  {set_name}:"
    print("{:10}".format(info_msg), end='')

    files = []
    for idx in idx_range:
        try:
            input_file = path.abspath(path.join(extracted_dir, set_name, "inputs", audio_prefix + str(idx).zfill(3) + ".wav"))

            if use_mirror_augment:
                assert(set_name == "train")
                # original files
                label_file = path.abspath(path.join(extracted_dir, set_name, "labels_mirrored", motion_prefix + str(idx).zfill(3) + "_mirrored.npz"))
            else:
                label_file = path.abspath(path.join(extracted_dir, set_name, "labels", motion_prefix + str(idx).zfill(3) + ".npz"))
            
            wav_size = path.getsize(input_file)
            files.append((input_file, wav_size, label_file))
        except OSError:
            continue

        print(idx, end=' ')

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "bvh_filename"])


def check_dataset_directories(raw_data_dir, use_mirror_augment):
    """
    Verify that 'raw_data_dir' exists and that it contains the 
    'Audio', 'Transcripts' and 'Motion' subdirectories.
    """
    if not path.isdir(raw_data_dir):
        abs_path = path.abspath(raw_data_dir)

        print(f"ERROR: The given dataset folder for the raw data ({abs_path}) does not exist!")
        print("Please, provide the correct path to the dataset in the `-raw_data_dir` argument.")
        exit(-1)


    data_directories = [ path.join(raw_data_dir, "Audio"),
        path.join(raw_data_dir, "Transcripts"),
        path.join(raw_data_dir, "Motion")] 
    
    if use_mirror_augment:
        data_directories.append(path.join(raw_data_dir, "Motion_mirrored"))
        
    for sub_dir in data_directories:
        if not path.isdir(sub_dir):
            _, name = path.split(sub_dir)
            print(f"ERROR: The '{name}' directory is missing from the given dataset folder: '{raw_data_dir}'!") 
            exit(-1)

if __name__ == "__main__":
    args = dataset_argparser.parse_args()
    
    check_dataset_directories(args.raw_data_dir, args.use_mirror_augment) 
    create_dataset_splits(args.raw_data_dir, args.proc_data_dir, args.use_mirror_augment)
    
    print(f"\nFinished!")

