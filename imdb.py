import numpy as np
import os
import shutil
import tempfile
import urllib.request

sep = os.path.sep

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
OUTPUT_NAME = "aclImdb"

N_SAMPLES = min((20172998 - 20000000)//10, 12500)

def download_and_extract_imdb():
    if os.path.exists(OUTPUT_NAME):
        print("Imdb dataset download target exists at " + OUTPUT_NAME)
    else:
        print("Downloading IMDB dataset...")
        with urllib.request.urlopen(IMDB_URL) as response:
            with tempfile.NamedTemporaryFile() as temp_archive:
                temp_archive.write(response.read())
                imdb_tar = shutil.unpack_archive(
                    temp_archive.name, extract_dir=".", format="gztar")

    return

def load_imdb():
    X_train = []
    y_train = []

    path = os.path.join('aclImdb', 'train', 'pos', '')
    X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_train.extend([1 for _ in range(N_SAMPLES)])

    path = os.path.join('aclImdb', 'train', 'neg', '')
    X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_train.extend([0 for _ in range(N_SAMPLES)])

    X_test = []
    y_test = []

    path = os.path.join('aclImdb', 'test', 'pos', '')
    X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([1 for _ in range(N_SAMPLES)])

    path = os.path.join('aclImdb', 'test', 'neg', '')
    X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([0 for _ in range(N_SAMPLES)])

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    return (X_train, y_train), (X_test, y_test)
