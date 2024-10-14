import asyncio
import concurrent.futures
import librosa
import numpy as np
from librosa.feature import chroma_cens
import os

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
from tqdm import tqdm
import gzip
import rp_extract as rp
import pickle
from sklearn.mixture import GaussianMixture

# import numpy as np
# import multiprocessing

# from torchvision import transforms
# import math
# import warnings
# import os
# import sys

# import torch
# import torch.nn as nn
# import gzip

# from tqdm import tqdm

# from operator import itemgetter 
# from ast import literal_eval
# import pickle

# import random
# import itertools
# from pytube import extract
# import librosa

# from torchvision import transforms

# from nnAudio import Spectrogram
# import essentia.standard as estd

# from resnet_ibn import resnet50_ibn_a
# from copy import deepcopy
# import math
# import time



sampling_rate = 16000
desired_length = 7717500

segment_duration = 0.4
overlap_duration = 0.1
# before 0.4 0.1
segment_length = int(segment_duration * sampling_rate)
overlap_length = int(overlap_duration * sampling_rate)


def split_list(input_list, n):
    return [input_list[i:i + n] for i in range(0, len(input_list), n)]

def load_song_as_cqt(mp3, mean_size):
    y = estd.MonoLoader(filename=f"{mp3}", sampleRate=22050)()
    if y.shape[0] < 22050 * 181:
        y = librosa.util.fix_length(y, size=22050 * 181)
    audio_vector = y[:int(22050 * 180)]
    cqt = cqt_layer(torch.Tensor(audio_vector)).squeeze()
    assert cqt.shape[0] == 84
    assert len(cqt.shape) == 2
    cqt = cqt.numpy()
    # add averaging as in ByteCover
    height, length = cqt.shape
    new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
    for j in range(int(length/mean_size)):
        new_cqt[:,j] = cqt[:,j*mean_size:(j+1)*mean_size].mean(axis=1)

    return (mp3, new_cqt)

def monitor_results(results):
    num_tasks = len(results)
    num_completed = 0

    while num_completed < num_tasks:
        num_completed = sum(1 for r in results if r.ready())
        print(f"Completed: {num_completed}/{num_tasks} tasks", end="\r")
        time.sleep(0.1)  # Adjust the polling interval as needed

    print("\nAll tasks completed.")
    
def extract_features_rp(mp3):
    sampling_rate = 44100
    sample_rate = sampling_rate
    audio_vector, sampling_rate_1 = librosa.load(mp3, sr = None)

    if sampling_rate_1 != sampling_rate:
        audio_vector = librosa.resample(audio_vector, orig_sr=sampling_rate_1, target_sr=sampling_rate)
    audio_vector = librosa.util.fix_length(audio_vector, size=sampling_rate * 181)
    wave_data = audio_vector[:int(sampling_rate * 180)]
    
    # wave_data, sample_rate = librosa.core.load(mp3, 
    #                                            sr    = 44100, 
    #                                            mono  = True, 
    #                                            dtype = np.float32)
    # wave_data = wave_data[:int(sample_rate * 180)]
    # normalize input amplitude
    wave_data = librosa.util.normalize(wave_data)
    
    # extract features
    feat = rp.rp_extract(wave_data,
                         sample_rate,
                         extract_rp          = True,
                         extract_ssd         = True,
                         extract_rh          = True,
                         extract_mvd         = True,
                         extract_tssd        = True,
                         extract_trh         = True,
                         n_bark_bands        = 24,
                         spectral_masking    = True,
                         transform_db        = True,
                         transform_phon      = True,
                         transform_sone      = True,
                         fluctuation_strength_weighting = True,
                         skip_leadin_fadeout = 0,
                         step_width          = 1,
                         mod_ampl_limit      = 60,
                         verbose             = False)

    return feat

def generate_gmm(mean_list, cov_list, weights, num_samples):
    samples = []
    for i in range(len(weights)):
        num_samples_i = int(num_samples * weights[i])
        samples_i = np.random.multivariate_normal(mean_list[i], cov_list[i], num_samples_i)
        samples.extend(samples_i)
    return np.array(samples)


def extract_features_gmm(mp3):
    audio_file = mp3
    
    sr = 16000 if 'covers80' in mp3 else 44100
    
    print('process 1')

    audio_signal, sample_rate = librosa.core.load(audio_file, 
                                            sr    = sr, 
                                            mono  = True, 
                                            dtype = np.float32)

    # Split audio into 25 ms frames
    frame_duration_ms = 100
    hop_duration_ms = 40
    print('process 2')
    frame_sample_length = int(sample_rate * (frame_duration_ms / 1000))
    hop_sample_length = int(sample_rate * (hop_duration_ms / 1000))
    frames = librosa.util.frame(audio_signal, frame_length=frame_sample_length, hop_length=frame_sample_length - hop_sample_length).T

    # Calculate MFCCs for each frame
    print('process 3')
    n_mfcc = 40
    mfccs = librosa.feature.mfcc(y=frames, sr=sample_rate, n_mfcc=n_mfcc, hop_length = 32)
    median_mfcc = np.median(mfccs, axis = 2)
    median_ref = median_mfcc[:, 1:]
    
    print('process 4')
    gm_ref = GaussianMixture(n_components=30, random_state=0).fit(median_ref)
    mean_ref = gm_ref.means_
    cov_ref = gm_ref.covariances_
    weight_ref = gm_ref.weights_
    num_samples = 1000
    samples_ref = generate_gmm(mean_ref, cov_ref, weight_ref, num_samples)
    
    return samples_ref


    
    
    


def process_audio_file(file_path):
    
    try:
        
        q = file_path.split('/')[-1]
        audio_vector, sampling_rate_1 = librosa.load(file_path, sr = None)
        # audio_vector = audio_vector[int(5*sampling_rate):int(sampling_rate * 180)]
        
        if sampling_rate_1 != sampling_rate:
            audio_vector = librosa.resample(audio_vector, orig_sr=sampling_rate_1, target_sr=sampling_rate)
        audio_vector = librosa.util.fix_length(audio_vector, size=sampling_rate * 181)
        audio_vector = audio_vector[:int(sampling_rate * 180)]
        
        # frame_hpcp = hpcpgram(audio_vector, 
        #                 sampleRate=sampling_rate, frameSize=segment_length, hopSize=segment_length - overlap_length)
        
        frame_cens = librosa.feature.chroma_cens(y=audio_vector, sr=sampling_rate, hop_length = overlap_length)


#         feature_gmm = extract_features_gmm(file_path)
#         f = open('/home/jupyter/rayhan_workdir/personal/perceptual-music-similarity-rayhan-ka/data/covers80/original/mfcc/' + q.replace('.mp3', '') + '.gz', 'wb')
#         pickle.dump(feature_gmm, f)
#         f.close()
        
        # f = gzip.GzipFile('/home/jupyter/rayhan_workdir/personal/perceptual-music-similarity-rayhan-ka/data/covers80/original/hpcp/' + q.replace('.mp3', '') + '.gz', "w")
        # np.save(file=f, arr=frame_hpcp)
        # f.close()
    except Exception as e:
        print(e)
        return 'Failed'
    return 'Success'

async def process_audio_files_async(file_paths):
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, process_audio_file, file) for file in file_paths]
        results = await asyncio.gather(*tasks)
    return results

async def main(audio_files):
    results = await process_audio_files_async(audio_files)


if __name__ == '__main__':
    from tqdm import tqdm
    source_path = "/home/jupyter/rayhan_workdir/personal/perceptual-music-similarity-rayhan-ka/data/covers80/original/audio/"
    source_list = os.listdir(r"{}".format(source_path))
    audio_files = [source_path + x for x in source_list]
    
    for fl in tqdm(audio_files):
        process_audio_file(fl)
    
    # asyncio.run(main(audio_files))
