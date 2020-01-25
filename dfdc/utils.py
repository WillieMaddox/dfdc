import os
import random
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wav

import torch
from torch._six import int_classes as _int_classes
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau


EPS = np.finfo(np.float32).eps


def get_prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def pad_string(x, n):
    """
    Pad a numeric string with zeros such that the resulting string len is n.
    Args:
        x: numeric string, e.g. '6234'
        n: length of final string length, e.g. 7

    Returns:
    a zero padded string, e.g. '0006234'
    """
    padding = n - len(x)
    x_new = x if padding <= 0 else ''.join(['0' * padding, x])
    return x_new


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def get_hamming_distance(hash1, hash2, normalize=False, as_score=False):
    """
    The args should be the same datatype as the output type of opencv img_hash blockMeanHash.
    Order does not matter. i.e. hash1, hash2 will produce the same result as hash2, hash1.

    :param hash1: len 32 or 96 (3*32) ndarray of uint8
    :param hash2: len 32 or 96 (3*32) ndarray of uint8
    :param normalize: bool. If True, normalize the metric [0, 1]
    :param as_score: bool. flips the hamming metric. The larger the score, the more perfect the match.
    :return: float if normalize is True, uint8 otherwise
    """
    h1 = np.unpackbits(hash1)
    h2 = np.unpackbits(hash2)

    hamming_metric = np.sum(h1 ^ h2, dtype=np.int)
    hamming_metric = len(h1) - hamming_metric if as_score else hamming_metric
    hamming_metric = hamming_metric / len(h1) if normalize else hamming_metric

    return hamming_metric


def get_hamming_distance_array(hash1, hash2, normalize=False, as_score=False, axis=1):
    """
    The args should be the same datatype as the output type of opencv img_hash blockMeanHash.
    Order does not matter. i.e. hash1, hash2 will produce the same result as hash2, hash1.

    :param hash1: len 32 ndarray of uint8
    :param hash2: len 32 ndarray of uint8
    :param normalize: bool. If True, normalize the metric [0, 1]
    :param as_score: bool. flips the hamming metric. The larger the score, the more perfect the match.
    :return: float if normalize is True, uint8 otherwise
    """
    h1 = np.unpackbits(hash1, axis=axis)
    h2 = np.unpackbits(hash2, axis=axis)

    hamming_metric = np.sum(h1 ^ h2, dtype=np.int, axis=axis)
    hamming_metric = h1.shape[-1] - hamming_metric if as_score else hamming_metric
    hamming_metric = hamming_metric / h1.shape[-1] if normalize else hamming_metric

    return hamming_metric


def fuzzy_join(tile1, tile2):
    maxab = np.max(np.stack([tile1, tile2]), axis=0)
    a = maxab - tile2
    b = maxab - tile1
    return a + b


def fuzzy_diff(tile1, tile2):
    ab = fuzzy_join(tile1, tile2)
    return np.sum(ab)


def fuzzy_norm(tile1, tile2):
    ab = fuzzy_join(tile1, tile2)
    n = 255 * np.sqrt(np.prod(ab.shape))
    return np.linalg.norm(255 - ab) / n


def fuzzy_compare(tile1, tile2):
    ab = fuzzy_join(tile1, tile2)
    n = 255 * np.prod(ab.shape)
    return np.sum(255 - ab) / n


def bce_loss(ytrue, yprob):
    return -1 * (np.log(np.max([EPS, yprob])) if ytrue else np.log(np.max([EPS, 1 - yprob])))


def even_split(n_samples, batch_size, split):
    # split the database into train/val sizes such that
    # batch_size divides them both evenly.
    # Hack until I can figure out how to ragged end of the database.
    n_batches = n_samples // batch_size
    n_train_batches = round(n_batches * split)
    n_valid_batches = n_batches - n_train_batches
    n_train = n_train_batches * batch_size
    n_valid = n_valid_batches * batch_size
    assert n_train + n_valid <= n_samples, n_train
    return n_train, n_valid


def add_tuples(t1, t2):
    return t1[0] + t2[0], t1[1] + t2[1]


class CSVLogger:

    def __init__(self, filename, header):
        self.filename = filename
        self.header = header

    def on_epoch_end(self, stats):

        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as ofs:
                ofs.write(','.join(self.header) + '\n')

        with open(self.filename, 'a') as ofs:
            ofs.write(','.join(map(str, stats)) + '\n')


class ReduceLROnPlateau2(ReduceLROnPlateau):

    def __init__(self, *args, **kwargs):
        super(ReduceLROnPlateau2, self).__init__(*args, **kwargs)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class SubsetSampler(data.Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ImportanceSampler(data.Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        num_records (int): Total number of samples in the dataset.
        num_samples (int): Number of samples to draw from the dataset.
        batch_size (int): Size of mini-batch.

    """

    def __init__(self, num_fake_records, num_real_records, num_samples, batch_size):

        num_records = num_fake_records + num_real_records

        if not isinstance(num_records, _int_classes) or isinstance(num_records, bool) or num_records <= 0:
            raise ValueError('num_records should be a positive integral value, but got num_records={}'.format(num_records))
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError('num_samples should be a positive integral value, but got num_samples={}'.format(num_samples))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integral value, but got batch_size={}'.format(batch_size))
        if num_records < num_samples < batch_size:
            raise ValueError('num_samples must be less than num_records and greater than batch_size')
        if num_samples % batch_size != 0:
            raise ValueError(f'batch_size ({batch_size}) must divide num_samples ({num_samples}) evenly.')

        self.num_steps = 0
        self.num_epochs = 0
        self.num_fake_records = num_fake_records
        self.num_real_records = num_real_records
        self.num_records = num_records
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size
        self.batch_size = batch_size
        self.drop_last = True

        self.ages = np.zeros(num_records, dtype=int)
        self.visits = np.zeros(num_records, dtype=int)
        self.losses = np.ones(num_records)  #* 27.6310
        self.epoch_losses = np.ones(num_samples) * -1.0
        self._epoch_ages = None

        fake_indices = np.random.choice(self.num_fake_records, self.num_samples // 2, replace=False)
        real_indices = np.random.choice(self.num_real_records, self.num_samples // 2, replace=False)
        self.indices = np.hstack([fake_indices, real_indices + self.num_fake_records])
        np.random.shuffle(self.indices)
        self.sampler = SubsetSampler(self.indices)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    @property
    def epoch_ages(self):
        if self._epoch_ages is None:
            # plus 1 since we're always lagging behind by 1 gradient step.
            x = np.arange(self.num_batches)[::-1] + 1
            self._epoch_ages = np.repeat(x, self.batch_size)
            assert len(self._epoch_ages) == self.num_samples
        return self._epoch_ages

    def update(self, batch_losses):
        idx = self.num_steps * self.batch_size
        self.epoch_losses[idx:idx + self.batch_size] = batch_losses[:, 0]
        self.num_steps += 1

    def on_epoch_end(self):
        """Use losses, visits and ages to update weights for samples"""

        assert np.min(self.epoch_losses) >= 0, np.min(self.epoch_losses)
        # age all records by the number of batches seen this epoch.
        self.ages += self.num_batches
        # only update the sampled records since their ages got reset.
        self.ages[self.indices] = self.epoch_ages
        # increment visits for samples by one.
        self.visits[self.indices] += 1
        # update losses
        self.losses[self.indices] = self.epoch_losses
        self.num_epochs += 1

        # normalize
        norm_ages = self.ages / np.sum(self.ages)
        # log_ages = np.log(self.ages)
        # norm_log_ages = log_ages / np.sum(log_ages)

        non_visits = self.num_epochs - self.visits
        norm_non_visits = non_visits / np.sum(non_visits)

        norm_losses = self.losses / np.sum(self.losses)
        weights = norm_ages + norm_non_visits + norm_losses
        # weights = log_ages * (np.sum(self.losses) / np.sum(log_ages)) + self.losses
        # norm_weights = weights / np.sum(weights)

        visits = np.clip(self.visits, 1, self.num_epochs)
        # tfidf0 = self.losses * np.log(self.ages / visits)
        ucb0 = self.losses + 2 * np.sqrt(np.log(self.ages) / visits)
        # self.indices = np.argsort(ucb)[-self.num_samples:]
        # self.indices = np.random.choice(self.num_records, self.num_samples, replace=False, p=self.norm_weights)

        weights = ucb0
        fake_weights = weights[:self.num_fake_records]
        real_weights = weights[self.num_fake_records:]
        fake_indices = np.argsort(fake_weights)[::-1][:self.num_samples // 2]
        real_indices = np.argsort(real_weights)[::-1][:self.num_samples // 2]
        self.indices = np.hstack([fake_indices, real_indices + self.num_fake_records])

        # self.indices = np.argsort(weights)[::-1][:self.num_samples]
        np.random.shuffle(self.indices)

        self.sampler = SubsetSampler(self.indices)
        self.num_steps = 0
        self.epoch_losses *= -1.0

        # print(self.num_epochs)
        # print(self.ages)
        # print(non_visits)
        # print(self.losses)
        # mask = np.zeros((self.num_records,), dtype=int)
        # mask[self.indices] = 11
        # print(mask)


if __name__ == '__main__':

    import time
    from collections import namedtuple
    import pandas as pd

    from torch.optim import Adam
    from tqdm import trange

    from dfdc.model import save_checkpoint
    from dfdc.model import FDNet as Net

    from dfdc.dataset_utils import TrainMelSpectrogramDataset as Dataset

    audio_dir = "/media/Aorus/DATA/dfdc/mels/"
    # audio_dir = "/media/Aorus/DATA/dfdc/mels_128_433/"
    models_dir = "/home/maddoxw/PycharmProjects/dfdc/models/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    dataset_filename = '../full_dfdc_dataset.csv'
    full_dataset_filename = dataset_filename
    df = pd.read_csv(full_dataset_filename)
    full_dataset = list(zip(*[df[c].values.tolist() for c in df]))

    model_basename = 'net0'
    date_time = get_datetime_now()

    # Parameters
    n_mels = 128
    n_timesteps = 865
    trainval_split = 0.9
    sample_rate = 0.1
    batch_size = 128
    max_epochs = 120
    num_workers = 22
    best_loss = 9999.0
    learning_rate = 0.01
    print(date_time)

    fake_dataset = []
    real_dataset = []
    for rec in full_dataset:
        if rec[1] == 1:
            fake_dataset.append(rec)
        else:
            real_dataset.append(rec)

    print(len(fake_dataset), len(real_dataset), len(fake_dataset)/len(full_dataset))

    np.random.shuffle(fake_dataset)
    np.random.shuffle(real_dataset)

    n_batches = len(full_dataset) // batch_size
    n_valid = round(n_batches * (1 - trainval_split)) * batch_size
    n_train = len(full_dataset) - n_valid

    pivot = n_valid // 2
    val_dataset = real_dataset[:pivot] + fake_dataset[:pivot]
    np.random.shuffle(val_dataset)
    len(val_dataset)

    train_fake_dataset = fake_dataset[pivot:]
    train_real_dataset = real_dataset[pivot:]
    train_dataset = train_fake_dataset + train_real_dataset
    n_train_fake = len(train_fake_dataset)
    n_train_real = len(train_real_dataset)

    partition = {'train': train_dataset, 'valid': val_dataset}
    n_samples = batch_size * (int(round(n_train * sample_rate)) // batch_size)
    print(n_train, n_valid, n_samples)

    df = pd.DataFrame()
    df['img_id'] = pd.Series(partition['train'])
    df.to_csv(os.path.join(models_dir, f'{model_basename}.{date_time}.ids.csv'), index=False)

    sampler = ImportanceSampler(n_train_fake, n_train_real, n_samples, batch_size)

    loader_params = {
        'train': {'batch_sampler': sampler, 'num_workers': num_workers},
        'valid': {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': False}}

    audio_transforms = {
        'train': lambda x: torch.from_numpy(x),
        'valid': lambda x: torch.from_numpy(x),
    }

    audio_datasets = {x: Dataset(partition[x], x, audio_transforms[x], audio_dir) for x in ['train', 'valid']}

    # Generators
    generators = {x: data.DataLoader(audio_datasets[x], **loader_params[x]) for x in ['train', 'valid']}
    print(len(generators['train']), len(generators['valid']))

    model = Net(n_mels, n_timesteps, 32, 8)

    model.to(device)

    # loss = torch.nn.BCELoss()
    loss = torch.nn.BCEWithLogitsLoss()
    # sample_loss = torch.nn.BCELoss(reduction='none')
    sample_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    optimizer = Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau2(optimizer, verbose=True)

    header = ["epoch", "time", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "train_time", "val_time"]
    Stats = namedtuple('Stats', header)
    csv_filename = os.path.join(models_dir, f'{model_basename}.{date_time}.metrics.csv')

    logger = CSVLogger(csv_filename, header)

    start_time = time.time()

    # Loop over epochs
    for epoch in range(max_epochs):

    #     scheduler.step()

        # Training
        t0 = time.time()
        total_train_loss = 0
        total_train_acc = 0
        model.train()
        t = trange(len(generators['train']))
        train_iterator = iter(generators['train'])
        for i in t:
            t.set_description(f'Epoch {epoch + 1:>03d}')
            # Get next batch and push to GPU
            inputs, labels = train_iterator.next()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = loss(outputs, labels)
            train_loss.backward()
            optimizer.step()

            #Print statistics
            sampler.update(sample_loss(outputs, labels).data.cpu().numpy())
            total_train_loss += train_loss.data.item()
            y_pred = outputs > 0.5
            y_pred = y_pred.type_as(torch.cuda.FloatTensor())
            equality = labels == y_pred
            total_train_acc += equality.type_as(torch.FloatTensor()).numpy().mean()

            loss_str = f'{total_train_loss / (i + 1):.6f}'
            acc_str = f'{total_train_acc / (i + 1):.5f}'
            t.set_postfix(loss=loss_str, acc=acc_str)

        train_loss = total_train_loss / (i + 1)
        train_acc = total_train_acc / (i + 1)
        train_time = time.time() - t0

        # Validation
        t1 = time.time()
        total_val_loss = 0
        total_val_acc = 0
        t = trange(len(generators['valid']))
        valid_iterator = iter(generators['valid'])
        with torch.no_grad():
            model.eval()
            for i in t:
                t.set_description(f'Validation')
                # Get next batch and push to GPU
                inputs, labels = valid_iterator.next()
                inputs, labels = inputs.to(device), labels.to(device)

                val_outputs = model(inputs)
                val_loss = loss(val_outputs, labels)

                total_val_loss += val_loss.data.item()
                y_pred = val_outputs > 0.5
                y_pred = y_pred.type_as(torch.cuda.FloatTensor())
                equality = labels == y_pred
                total_val_acc += equality.type_as(torch.FloatTensor()).numpy().mean()

                loss_str = f'{total_val_loss / (i + 1):.6f}'
                acc_str = f'{total_val_acc / (i + 1):.5f}'
                t.set_postfix(loss=loss_str, acc=acc_str)

        val_loss = total_val_loss / (i + 1)
        val_acc = total_val_acc / (i + 1)
        val_time = time.time() - t1

        sampler.on_epoch_end()
        total_time = time.time() - start_time

        df = pd.DataFrame()
        df['ages'] = pd.Series(sampler.ages)
        df['visits'] = pd.Series(sampler.visits)
        df['losses'] = pd.Series(sampler.losses)
        df.to_csv(os.path.join(models_dir, f'{model_basename}.{date_time}.{epoch + 1:03d}.{val_loss:.6f}.avl.csv'), index=False)

        if val_loss < best_loss:
            save_checkpoint(os.path.join(models_dir, f'{model_basename}.{date_time}.{epoch + 1:03d}.{val_loss:.6f}.pth'), model)
            save_checkpoint(os.path.join(models_dir, f'{model_basename}.{date_time}.best.pth'), model)
            save_checkpoint(os.path.join(models_dir, f'{model_basename}.best.pth'), model)
            best_loss = val_loss

        stats = Stats(epoch+1, total_time, scheduler.get_lr()[0], train_loss, train_acc, val_loss, val_acc, train_time, val_time)
        logger.on_epoch_end(stats)

        scheduler.step(val_loss)