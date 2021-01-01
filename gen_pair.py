import torch
import numpy as np
import soundfile as sf
import librosa
import argparse
import glob
import os
import random
import h5py
import time
import math
from scipy.io import wavfile


def gen_train_mix(train_clean_name):
    print('Begin to generate mix utterance for training data')

    # path
    train_clean_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/TIMIT/train'
    train_noise_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/seen_long_noise.bin'
    train_mix_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/timit_mix_v2/trainset'

    # setting
    fs = 16000
    train_snr_list = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    mix_num_utterance = 20000
    #train_clean_list = glob.glob(os.path.join(train_clean_path, "*.wav"))

    # read noise_bin
    noise = np.memmap(train_noise_path, dtype=np.float32, mode='r')

    for count in range(mix_num_utterance):
        filename = '%s_%d' % ('train_mix', count+1)
        train_writer = h5py.File(train_mix_path + '/' + filename, 'w')

        clean_speech_id = random.randint(0, len(train_clean_name) - 1)
        train_snr_id = random.randint(0, len(train_snr_list) - 1)

        cln, sr = sf.read(os.path.join(train_clean_path, train_clean_name[clean_speech_id]))
        if sr != fs:
            raise ValueError('Invalid sample rate')

        train_snr = train_snr_list[train_snr_id]

        # choose the start of cutting noise
        noise_begin = random.randint(0, noise.size - cln.size)
        while np.sum(noise[noise_begin:noise_begin+cln.size] ** 2.0) == 0:
            noise_begin = random.randint(0, noise.size - cln.size)

        noise_cut = noise[noise_begin:noise_begin+cln.size]

        # mix
        alpha = np.sqrt(np.sum(cln ** 2.0) / (np.sum(noise_cut ** 2.0) * (10.0 ** (train_snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(cln ** 2.0) / (np.sum((noise_cut * alpha) ** 2.0)))
        mix = cln + alpha * noise_cut

        # energy normalization
        c = np.sqrt(1.0 * mix.size / np.sum(mix ** 2))
        mix = mix * c
        cln = cln * c

        # save h5py file
        train_writer.create_dataset('noisy_raw', data=mix.astype(np.float32), chunks=True)
        train_writer.create_dataset('clean_raw', data=cln.astype(np.float32), chunks=True)
        train_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    train_file_list = sorted(glob.glob(os.path.join(train_mix_path, '*')))
    read_train = open("train_file_list", "w+")

    for i in range(len(train_file_list)):
        read_train.write("%s\n" % (train_file_list[i]))

    read_train.close()
    print('making training data finished!')


def gen_val_mix(val_clean_name):
    print('Begin to generate mix utterance for validation data')

    # path
    val_clean_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/TIMIT/train'
    val_noise_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/noise/test_noise/babble.wav'
    val_mix_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/timit_mix_v2/valset'

    if not os.path.isdir(val_mix_path):
        os.makedirs(val_mix_path)

    # setting
    fs = 16000
    val_snr_list = -5.0
    mix_num_utterance = 200
    #val_clean_list = glob.glob(os.path.join(val_clean_path, "*.wav"))
    #val_clean_name = os.listdir(val_clean_path)

    # read noise
    noise, sr = sf.read(val_noise_path)
    if sr != 16000:
        raise ValueError('Invalid sample rate')

    for count in range(mix_num_utterance):
        filename = '%s_%d' % ('val_mix', count+1)
        val_writer = h5py.File(val_mix_path + '/' + filename, 'w')

        clean_speech_id = random.randint(0, len(val_clean_name) - 1)
        #val_snr_id = random.randint(0, len(val_snr_list) - 1)

        cln, sr = sf.read(os.path.join(val_clean_path, val_clean_name[clean_speech_id]))
        if sr != fs:
            raise ValueError('Invalid sample rate')

        val_snr = val_snr_list

        # choose the start of cutting noise
        noise_begin = random.randint(0, noise.size - cln.size)
        while np.sum(noise[noise_begin:noise_begin+cln.size] ** 2.0) == 0:
            noise_begin = random.randint(0, noise.size - cln.size)

        noise_cut = noise[noise_begin:noise_begin+cln.size]

        # mix
        alpha = np.sqrt(np.sum(cln ** 2.0) / (np.sum(noise_cut ** 2.0) * (10.0 ** (val_snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(cln ** 2.0) / (np.sum((noise_cut * alpha) ** 2.0)))
        mix = cln + alpha * noise_cut

        # energy normalization
        c = np.sqrt(1.0 * mix.size / np.sum(mix ** 2))
        mix = mix * c
        cln = cln * c

        # save h5py file
        val_writer.create_dataset('noisy_raw', data=mix.astype(np.float32), chunks=True)
        val_writer.create_dataset('clean_raw', data=cln.astype(np.float32), chunks=True)
        val_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    val_file_list = sorted(glob.glob(os.path.join(val_mix_path, '*')))
    read_train = open("validation_file_list", "w+")

    for i in range(len(val_file_list)):
        read_train.write("%s\n" % (val_file_list[i]))

    read_train.close()
    print('making validation data finished!')


def gen_test_mix():
    print('Begin to generate mix utterance for unseen testing data')

    # path
    test_clean_path = '/data/datasets/TIMIT/test'

    test_noise_path = '/data/datasets/noise/noisex92'
    #test_noise_path = '/data/KaiWang/pytorch_learn/speech_project/datasets/test_noise'

    #test_mix_path_noisy = '/data/KaiWang/pytorch_learn/speech_project/datasets/timit_mix/testset/noisy_test'
    #test_mix_path_clean = '/data/KaiWang/pytorch_learn/speech_project/datasets/timit_mix/testset/clean_test'
    test_mix = '/data/KaiWang/pytorch_learn/speech_project/datasets/timit_mix/testset_v1'

    test_noise_type = ['babble.wav']
    test_snr_list = [-5.0]

    test_clean_list = glob.glob(os.path.join(test_clean_path, "*.wav"))
    #test_clean_list1 = test_clean_list[:60]

    for idx, noise_name in enumerate(test_noise_type):

        print('Using %s noise' % (noise_name[:-4]))

        # read noise
        noise, sr = sf.read(os.path.join(test_noise_path, noise_name))

        if len(noise.shape) == 2:
            noise = noise[:, 0]

        if sr != 16000:
            raise ValueError('Invalid sample rate')

        for i, snr in enumerate(test_snr_list):
            print('SNR level: %d dB' % snr)

            for file_index, file_dir in enumerate(test_clean_list):

                filename = '%s_%d' % ('test_mix', file_index + 1)
                test_writer = h5py.File(test_mix + '/' + filename, 'w')

                #cln, srate = sf.read(file_dir)
                srate, cln = wavfile.read(file_dir)
                if srate != 16000:
                    raise ValueError('Invalid sample rate')

                if len(noise) < len(cln):
                    raise TypeError('length is not matched')


                # choose a start point for cutting noise
                noise_begin = random.randint(0, noise.size - cln.size)

                while np.sum(noise[noise_begin:noise_begin + cln.size] ** 2.0) == 0:
                    noise_begin = random.randint(0, noise.size - cln.size)

                noise_cut = noise[noise_begin:noise_begin + cln.size]

                # mix
                alpha = np.sqrt(np.sum(cln ** 2.0) / (np.sum(noise_cut ** 2.0) * (10.0 ** (snr / 10.0))))
                snr_check = 10.0 * np.log10(np.sum(cln ** 2.0) / (np.sum((noise_cut * alpha) ** 2.0)))
                mix = cln + alpha * noise_cut

                # energy normalization
                c = np.sqrt(1.0 * mix.size / np.sum(mix ** 2))
                mix = mix * c
                cln = cln * c

                #cln, mix, moise_wav = get_noisy_wav(cln, noise, snr=snr)


                test_writer.create_dataset('noisy_raw', data=mix.astype(np.float32), chunks=True)
                test_writer.create_dataset('clean_raw', data=cln.astype(np.float32), chunks=True)
                test_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    test_file_list = sorted(glob.glob(os.path.join(test_mix, '*')))
    read_train = open("test_file_list", "w+")

    for i in range(len(test_file_list)):
        read_train.write("%s\n" % (test_file_list[i]))

    read_train.close()

    print('making testing data finished!')



if __name__ == "__main__":

    train_clean_path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/TIMIT/train'
    train_clean_name = sorted(os.listdir(train_clean_path))
    print(len(train_clean_name))
    val_len = math.ceil(len(train_clean_name) * 0.05)
    start = random.randint(0, len(train_clean_name) - val_len)
    val_clean_name = train_clean_name[start:start + val_len]
    print(len(val_clean_name))
    tra_clean_name = [x for x in train_clean_name if x not in val_clean_name]
    print(len(tra_clean_name))

    gen_train_mix(tra_clean_name)
    gen_val_mix(val_clean_name)
    #gen_test_mix()



