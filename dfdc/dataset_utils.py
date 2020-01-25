import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import librosa
import torch
from torch.utils import data


def encode_mu_law(x, mu=256):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.long)


def decode_mu_law(y, mu=256):
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x


def sine_generator(seq_size=6000, mu=256):
    framerate = 44100
    t = np.linspace(0, 5, framerate * 5)
    data = np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 224 * t)
    data = data / 2
    while True:
        start = np.random.randint(0, data.shape[0] - seq_size)
        ys = data[start:start + seq_size]
        ys = encode_mu_law(ys, mu)
        yield torch.from_numpy(ys[:seq_size])


def get_dummy_dataset():
    fs = 44100
    interval = 0.2
    fmin = 200
    fmax = fs // 2
    # fmin = 50
    # fmax = 14000
    t = np.linspace(0, interval, int(fs * interval))
    X = np.zeros(((fmax - fmin) // 2 * 10, int(fs * interval)), np.float32)
    Y = np.zeros((fmax - fmin) // 2 * 10, np.float32)
    counter = 0
    for k in tqdm(range(fmin, fmax, 2)):  # Creating signals with different frequencies
        for phi in np.arange(0, 1, 0.1):
            X[counter] = np.sin(2 * np.pi * (k * t + phi))
            Y[counter] = k / interval
            counter += 1
    return X, Y


def search_for_record(filename0, train_dir):
    for dirname, _, filenames in os.walk(train_dir):
        for filename in filenames:
            if filename == filename0:
                return os.path.join(dirname, filename)


def fraction2float(s):
    """
    Converts a string like '2995/100' to a float 29.95

    Args:
        s: a string

    Returns:

    a float
    """
    num, denom = s.split('/')
    return 0 if denom == '0' else float(num) / float(denom)


def fraction2tuple(s):
    """
    Converts a string like '2995/100' to a tuple of ints (2995, 100)

    Args:
        s: a string

    Returns:

    a tuple of 2 ints
    """
    num, denom = s.split('/')
    return int(num), int(denom)


def parse_ffprobe_metadata(meta_file):
    metadata = {}
    with open(meta_file) as ifs:
        raw_metadata = json.load(ifs)

    stream_convert = {
        'codec_type': str,
        'bit_rate': int,
        'time_base': fraction2float,
        'sample_rate': int,
        'avg_frame_rate': fraction2float,
        'start_pts': int,
        'start_time': float,
        'duration_ts': int,
        'duration': float,
        'size': int,
        'width': int,
        'height': int,
        'channels': int,
        'nb_frames': int}

    meta_format = {}
    for key, value in raw_metadata['format'].items():
        if key in stream_convert:
            meta_format[key] = stream_convert[key](value)
    metadata['format'] = meta_format
    for raw_stream in raw_metadata['streams']:
        meta_stream = {}
        for key, value in raw_stream.items():
            if key in stream_convert:
                meta_stream[key] = stream_convert[key](value)
        codec_type = meta_stream.pop('codec_type')
        metadata[codec_type] = meta_stream
    return metadata


def flatten_ffprobe_metadata(meta_file):

    format_keep = (
        # 'disposition',
        # 'tags',
        # 'filename',
        # 'nb_streams',
        # 'nb_programs',
        # 'format_name',
        # 'format_long_name',
        'start_time',
        'duration',
        'size',
        'bit_rate',
        # 'probe_score',
    )

    stream_keep = {
        'video': (
            # 'disposition',
            # 'tags',
            # 'index',
            # 'codec_name',
            # 'codec_long_name',
            # 'profile',
            # 'codec_time_base',
            # 'codec_tag_string',
            # 'codec_tag',
            'width',
            'height',
            # 'coded_width',
            # 'coded_height',
            # 'has_b_frames',
            # 'sample_aspect_ratio',
            'display_aspect_ratio',
            # 'pix_fmt',
            # 'level',
            # 'chroma_location',
            # 'refs',
            # 'is_avc',
            # 'nal_length_size',
            'r_frame_rate',
            # 'avg_frame_rate',
            # 'time_base',
            # 'start_pts',
            # 'start_time',
            'duration_ts',
            'duration',
            # 'bit_rate',
            # 'bits_per_raw_sample',
            'nb_frames',
        ),
        'audio': (
            # 'disposition',
            # 'tags',
            # 'index',
            # 'codec_name',
            # 'codec_long_name',
            # 'profile',
            # 'codec_time_base',
            # 'codec_tag_string',
            # 'codec_tag',
            # 'sample_fmt',
            'sample_rate',
            # 'channels',
            # 'channel_layout',
            # 'bits_per_sample',
            # 'avg_frame_rate',
            # 'r_frame_rate',
            # 'time_base',
            'start_pts',
            'start_time',
            'duration_ts',
            'duration',
            # 'bit_rate',
            # 'max_bit_rate',
            'nb_frames',
        )
    }

    audio_stub = {
        'disposition': {},
        'tags': {},
        'index': '1',
        'codec_name': '',
        'codec_long_name': '',
        'profile': '',
        'codec_type': 'audio',
        'codec_time_base': '1/0',
        'codec_tag_string': '',
        'codec_tag': '',
        'sample_fmt': '',
        'sample_rate': '0',
        'channels': '0',
        'channel_layout': '',
        'bits_per_sample': '0',
        'r_frame_rate': '0/0',
        'avg_frame_rate': '0/0',
        'time_base': '1/0',
        'start_pts': '0',
        'start_time': '0',
        'duration_ts': '0',
        'duration': '0',
        'bit_rate': '0',
        'max_bit_rate': '0',
        'nb_frames': '0',
    }

    stream_convert = {
        'codec_type': str,
        'max_bit_rate': int,
        'bit_rate': int,
        'codec_time_base': fraction2tuple,
        'sample_rate': int,
        'r_frame_rate': fraction2tuple,
        'avg_frame_rate': fraction2tuple,
        'time_base': fraction2tuple,
        'start_pts': int,
        'start_time': float,
        'duration_ts': int,
        'duration': float,
        'size': int,
        'width': int,
        'height': int,
        'channels': int,
        'nb_frames': int}

    metadata = {}
    with open(meta_file) as ifs:
        raw_metadata = json.load(ifs)

    for key, value in raw_metadata['format'].items():
        if key not in format_keep:
            continue
        new_key = '_'.join(['format', key])
        if key in stream_convert:
            metadata[new_key] = stream_convert[key](value)
        else:
            metadata[new_key] = value

    raw_streams = raw_metadata['streams']
    if len(raw_streams) == 1:
        raw_streams.append(audio_stub)

    for raw_stream in raw_streams:
        codec_type = raw_stream.pop('codec_type')
        for key, value in raw_stream.items():
            if key not in stream_keep[codec_type]:
                continue
            new_key = '_'.join([codec_type, key])
            if key in stream_convert:
                metadata[new_key] = stream_convert[key](value)
            else:
                metadata[new_key] = value

    return metadata


def create_metadata_master_raw(data_path):
    label = {'FAKE': 1, 'REAL': 0}
    ffprobe_dir = os.path.join(data_path, 'metadata', 'ffprobe')
    metadata_master = {}
    for ii in range(50):
        mp4_dir = f'dfdc_train_part_{ii}'
        metadata_file = os.path.join(data_path, mp4_dir, 'metadata.json')
        with open(metadata_file) as ifs:
            label_metadata = json.load(ifs)
        for key, meta in label_metadata.items():
            ffprobe_file = os.path.join(ffprobe_dir, key.replace('.mp4', '.json'))
            if not os.path.exists(ffprobe_file):
                continue
            meta['label'] = label[meta['label']]
            # meta['mp4_dir'] = mp4_dir
            meta['zipfile_id'] = ii
            ffprobe_meta = flatten_ffprobe_metadata(ffprobe_file)
            # with open(ffprobe_file) as ifs:
            #     raw_metadata = json.load(ifs)
            # meta['ffprobe'] = raw_metadata
            meta = {**meta, **ffprobe_meta}
            metadata_master[key] = meta
    return metadata_master


def create_metadata_master(data_path):
    label = {'FAKE': 1, 'REAL': 0}
    ffprobe_dir = os.path.join(data_path, 'metadata', 'ffprobe')
    metadata_master = {}
    for ii in range(50):
        mp4_dir = f'dfdc_train_part_{ii}'
        metadata_file = os.path.join(data_path, mp4_dir, 'metadata.json')
        with open(metadata_file) as ifs:
            label_metadata = json.load(ifs)
        for key, meta in label_metadata.items():
            ffprobe_file = os.path.join(ffprobe_dir, key.replace('.mp4', '.json'))
            if not os.path.exists(ffprobe_file):
                continue
            meta['zipfile_id'] = ii
            meta['filename_npy'] = key.replace('.mp4', '.npy')
            if 'original' in meta:
                meta['original_npy'] = meta['original'].replace('.mp4', '.npy')
            meta['ffprobe'] = parse_ffprobe_metadata(ffprobe_file)
            meta['label'] = label[meta['label']]
            meta['mp4_dir'] = mp4_dir
            metadata_master[key] = meta
    return metadata_master


def load_all_metadata(data_path, audio_dir='wavs', mels_dir='mels'):
    label = {'FAKE': 1, 'REAL': 0}
    metadata_main = {}
    for ii in range(50):
        ii_has_missing = False
        mp4_dir = f'dfdc_train_part_{ii}'
        metadata_file = os.path.join(data_path, mp4_dir, 'metadata.json')
        with open(metadata_file) as ifs:
            metadata = json.load(ifs)
        for mp4_filename, meta in metadata.items():
            mp4_file_is_missing = False
            new_audio_file_is_missing = False
            new_audio_orig_is_missing = False
            has_missing = False
            mp4_file = os.path.join(data_path, mp4_dir, mp4_filename)
            if not os.path.exists(mp4_file):
                mp4_file_is_missing = True
                has_missing = True
                ii_has_missing = True
            new_filename = mp4_filename.replace('.mp4', '.npy')
            new_audio_file = os.path.join(data_path, audio_dir, new_filename)
            if not os.path.exists(new_audio_file):
                new_audio_file_is_missing = True
                has_missing = True
                ii_has_missing = True
            new_mel_file = os.path.join(data_path, mels_dir, new_filename)
            if not os.path.exists(new_mel_file):
                new_mel_file_is_missing = True
                has_missing = True
                ii_has_missing = True
            if label[meta['label']]:
                new_original = meta['original'].replace('.mp4', '.npy')
                new_audio_orig = os.path.join(data_path, audio_dir, new_original)
                if not os.path.exists(new_audio_orig):
                    new_audio_orig_is_missing = True
                    has_missing = True
                    ii_has_missing = True
                new_mel_orig = os.path.join(data_path, mels_dir, new_original)
                if not os.path.exists(new_mel_orig):
                    new_mel_orig_is_missing = True
                    has_missing = True
                    ii_has_missing = True
            if has_missing:
                print(f'{ii:>2}', end=' ')
                print(f'{mp4_filename} {mp4_file_is_missing:>1}', end=' ')
                print(f'{new_filename} {new_audio_file_is_missing:>1} {meta["label"]}', end=' ')
                print(f'{new_filename} {new_mel_file_is_missing:>1} {meta["label"]}', end=' ')
                if label[meta['label']]:
                    print(f'{new_original} {new_audio_orig_is_missing:>1} REAL', end=' ')
                    print(f'{new_original} {new_mel_orig_is_missing:>1} REAL', end=' ')
                print('')
                continue

            meta['original_npy'] = new_original
            meta['label'] = label[meta['label']]
            meta['mp4_dir'] = mp4_dir
            metadata_main[new_filename] = meta
        if ii_has_missing:
            print('')
    return metadata_main


def create_audio_waveform_dataset(data_dir, audio_topdir='mels'):
    label = {'FAKE': 1, 'REAL': 0}
    metadata_main = []
    for ii in range(50):
        mp4_dir = f'dfdc_train_part_{ii}'
        metadata_file = os.path.join(data_dir, mp4_dir, 'metadata.json')
        with open(metadata_file) as ifs:
            metadata = json.load(ifs)
        for key, meta in metadata.items():
            new_key = key.replace('.mp4', '.npy')
            audio_file = os.path.join(data_dir, audio_topdir, new_key)
            if not os.path.exists(audio_file):
                continue
            metadata_main.append((new_key, label[meta['label']]))
    return metadata_main


class TrainWaveFormDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, records, train_or_valid, audio_transforms, audio_dir):
        """Initialization"""
        self.records = records
        self.audio_dir = audio_dir
        self.valid = train_or_valid == 'valid'
        self.audio_transforms = audio_transforms
        self.seq_len = 221088

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.records)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.process_record(self.records[index])  # X, y

    def process_record(self, record):
        fname, is_fake = record
        X = np.zeros((self.seq_len,), dtype=np.float32)
        af = np.load(os.path.join(self.audio_dir, fname))
        n_samps = len(af)
        X[:n_samps] = af[:self.seq_len]
        X = X.reshape((1, -1))
        # print(X.shape, type(X[0, 0]))
        X = self.audio_transforms(X)
        # print(X.shape, type(X[0, 0]))
        y = np.array([is_fake], dtype=np.float32)
        return X, y


class TrainMelSpectrogramDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, records, train_or_valid, audio_transforms, audio_dir):
        """Initialization"""
        self.records = records
        self.mel_dir = audio_dir
        self.valid = train_or_valid == 'valid'
        self.audio_transforms = audio_transforms

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.records)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.process_record(self.records[index])  # X, y

    def process_record(self, record):
        fname, is_fake = record
        X = np.load(os.path.join(self.mel_dir, fname))
        X = self.audio_transforms(X)
        y = np.array([is_fake], dtype=np.float32)
        return X, y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(b)
