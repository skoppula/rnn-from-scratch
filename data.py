import csv
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split

SMS_FILENAME = 'data/sms/sms.txt'
SHAKESPEARE_FILENAME = 'data/shakespeare/shakespeare.txt'
PAULG_FILENAME = 'data/paulg/paulg.txt'

SHAKESPEARE_PATH = 'data/shakespeare/'
SMS_PATH = 'data/sms/'
PAULG_PATH = 'data/paulg/'

def read_lines_sms(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return [ row[-1] for row in list(reader) ]

def read_lines(filename):
    with open(filename) as f:
        return f.read().split('\n')

def index_(lines):
    vocab = list(set('\n'.join(lines)))
    ch2idx = { k:v for v,k in enumerate(vocab) }
    return vocab, ch2idx

def to_array(lines, seqlen, ch2idx):
    raw_data = '\n'.join(lines)
    num_chars = len(raw_data)
    data_len = num_chars//seqlen

    X = np.zeros([data_len, seqlen])
    Y = np.zeros([data_len, seqlen])

    for i in range(0, data_len):
        X[i] = np.array([ ch2idx[ch] for ch in raw_data[i*seqlen:(i+1)*seqlen] ])
        Y[i] = np.array([ ch2idx[ch] for ch in raw_data[(i*seqlen) + 1 : ((i+1)*seqlen) + 1] ])

    return X.astype(np.int32), Y.astype(np.int32)

def process_data(path, filename, seqlen=30):
    lines = read_lines(filename)
    idx2ch, ch2idx = index_(lines)

    X, y = to_array(lines, seqlen, ch2idx)
    X_trn, X_test, y_trn, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.save(path + 'trn_X.npy', X_trn)
    np.save(path + 'trn_y.npy', y_trn)
    np.save(path + 'test_X.npy', X_test)
    np.save(path + 'test_y.npy', y_test)

    with open(path+ 'metadata.pkl', 'wb') as f:
        pkl.dump( {'idx2ch' : idx2ch, 'ch2idx' : ch2idx }, f )

def load_data(path):
    with open(path + 'metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)

    X_trn = np.load(path + 'trn_X.npy')
    y_trn = np.load(path + 'trn_y.npy')
    X_test = np.load(path + 'test_X.npy')
    y_test = np.load(path + 'test_y.npy')

    return X_trn, y_trn, X_test, y_test, metadata['idx2ch'], metadata['ch2idx']

if __name__ == '__main__':
    process_data(path = PAULG_PATH, filename = PAULG_FILENAME)

