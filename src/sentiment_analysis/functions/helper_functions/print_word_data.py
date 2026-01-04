import numpy as np

def print_first_items_from_npy(npy_file_path, num_items=5):
    """
    Loads a .npy file containing a dictionary or ndarray and prints the first few items.

    Args:
        npy_file_path (str): Path to the .npy file.
        num_items (int): Number of items to print.
    """
    data = np.load(npy_file_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():  # 0-d array (likely a dict)
        data = data.item()
    # if isinstance(data, dict):
        # for i, (key, value) in enumerate(data.items()):
        #     print(f"{i+1}. Key: {key}")
        #     print(f"   Value: {value}")
        #     if i + 1 >= num_items:
        #         break
    if isinstance(data, np.ndarray):
        print(f"Loaded ndarray of shape {data.shape} and dtype {data.dtype}")
        print(f"Number of dimensions: {data.ndim}")
        # If ndarray contains tuples or objects, print key-value style if possible
        # for i in range(min(num_items, data.shape[0])):
        #     item = data[i]
        #     if isinstance(item, (tuple, list)) and len(item) == 2:
        #         key, value = item
        #         print(f"{i+1}. Key: {key}")
        #         print(f"   Value: {value}")
        #     else:
        #         print(f"{i+1}. Value: {item}")
    else:
        print("Loaded object is not an ndarray. Type:", type(data))

# Example usage:
if __name__ == "__main__":
    file_path = 'testing_scrap_misc/scrap_01/cooccurrence_probability_matrix.npy'
    print_first_items_from_npy(file_path, num_items=5)