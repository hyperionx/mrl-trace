import h5py

def print_h5_tree(obj, level=0):
    for key, val in obj.items():
        print("  " * level + str(key) + " : " + str(type(val)))
        if isinstance(val, h5py.Group):
            print_h5_tree(val, level + 1)
        elif isinstance(val, h5py.Dataset):
            print("  " * (level + 1) + f"Shape: {val.shape}, Type: {val.dtype}")

with h5py.File("test.nwb", "r") as f:
    print_h5_tree(f)
