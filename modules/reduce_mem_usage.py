import numpy as np
import pandas as pd


def reduce_mem_usage(data: pd.DataFrame) -> None:
    """Function to optimize DataFrame memory usage (inplace)."""

    # Calculating initial memory usage.
    start_memory = data.memory_usage().sum() / 1024 ** 2
    print(f"Initial memory usage: {start_memory:.2f} MB")

    # Creating dictionaries with ranges for each number type.
    int_type_dict = {
        (np.iinfo(np.int8).min, np.iinfo(np.int8).max): np.int8,
        (np.iinfo(np.int16).min, np.iinfo(np.int16).max): np.int16,
        (np.iinfo(np.int32).min, np.iinfo(np.int32).max): np.int32,
        (np.iinfo(np.int64).min, np.iinfo(np.int64).max): np.int64,
    }

    float_type_dict = {
        #         (np.finfo(np.float16).min, np.finfo(np.float16).max): np.float16,
        (np.finfo(np.float32).min, np.finfo(np.float32).max): np.float32,
        (np.finfo(np.float64).min, np.finfo(np.float64).max): np.float64,
    }

    # Process each column in the DataFrame.
    for column in data.columns:
        col_type = data[column].dtype

        if np.issubdtype(col_type, np.integer):
            c_min = data[column].min()
            c_max = data[column].max()
            dtype = next((v for k, v in int_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                data[column] = data[column].astype(dtype)
        elif np.issubdtype(col_type, np.floating):
            c_min = data[column].min()
            c_max = data[column].max()
            dtype = next((v for k, v in float_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                data[column] = data[column].astype(dtype)

    # Calculation of final memory usage.
    end_memory = data.memory_usage().sum() / 1024 ** 2
    print(f"Final memory usage: {end_memory:.2f} MB")
    print(f"Reduced by {(start_memory - end_memory) / start_memory * 100:.1f}%")


if __name__ == '__main__':
    reduce_mem_usage()
