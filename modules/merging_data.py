import os
import pandas as pd
import gc
import pyarrow

from tqdm import tqdm
from modules.reduce_mem_usage import reduce_mem_usage


# Path to the folder with raw data.
path = 'raw_data/'


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                    num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    reads num_parts_to_read partitions, converts them to pd.DataFrame, and returns
    :param verbose: bool
    :param path_to_dataset: path to the partitions directory
    :param start_from: number of partition to start reading from
    :param num_parts_to_read: number of partitions to read
    :param columns: list of columns to be read from the partition
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                            if filename.startswith('train')])
    print(dataset_paths)

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm(chunks, desc="Reading dataset with pandas"):
        print('chunk_path', chunk_path)
        chunk = pd.read_parquet(chunk_path, columns=columns)
        reduce_mem_usage(chunk)
        res.append(chunk)

    return pd.concat(res).reset_index(drop=True)


def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1,
                                 num_parts_total: int = 50, verbose: bool = True) -> pd.DataFrame:
    """
    returns a finished pd.DataFrame with attributes on which to learn a model for the target task
    path_to_dataset: str
        path to dataset with partitions
    num_parts_to_preprocess_at_once: int
        number of partitions that will be held and processed in memory at the same time
    num_parts_total: int
        total number of partitions to be processed
    verbose: bool
        logs each part of data to be processed
    """
    preprocessed_frames = []

    for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                     desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once,
                                                             verbose=verbose)
        preprocessed_frames.append(transactions_frame)

    df = pd.concat(preprocessed_frames)

    gc.collect()

    return df
