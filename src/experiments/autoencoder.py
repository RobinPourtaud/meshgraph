from ..tools.dataset import load_MNISTSuperpixels

def load_data(conf):
    print("Loading the data...")
    return load_MNISTSuperpixels(conf["data_path"])


def train(conf):
    print("Training the model...")