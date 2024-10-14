from nnAudio import Spectrogram
import essentia.standard as estd

from copy import deepcopy
import math
import librosa
from simple_fast import batch_simple_fast_gpu, simple_fast, multi_simple_fast
from essentia.pytools.spectral import hpcpgram
import cupy as cp

import os
import gzip
import numpy as np
import gc

import pandas as pd
from operator import itemgetter 
from ast import literal_eval
from tqdm import tqdm
import time

def from_audio_to_features(mp3, feature = 'hpcp'):
    sampling_rate = 16000
    
    segment_duration = 0.4
    overlap_duration = 0.1

    segment_length = int(segment_duration * sampling_rate)
    overlap_length = int(overlap_duration * sampling_rate)

    audio_vector, sampling_rate_1 = librosa.load(mp3, sr = None)
    if sampling_rate_1 != sampling_rate:
        audio_vector = librosa.resample(audio_vector, orig_sr=sampling_rate_1, target_sr=sampling_rate)
    audio_vector = librosa.util.fix_length(audio_vector, size=sampling_rate * 181)
    audio_vector = audio_vector[:int(sampling_rate * 180)]
    
    if feature == 'hpcp':
        frame_hpcp = hpcpgram(audio_vector, 
                    sampleRate=sampling_rate, frameSize=segment_length, hopSize=segment_length - overlap_length)
        return frame_hpcp
    
    elif feature == 'mfcc':
        frame_mfcc = librosa.feature.mfcc(y=audio_vector, sr=sampling_rate, n_mfcc=20 ,n_fft = segment_length, hop_length = segment_length - overlap_length)
        frame_mfcc = frame_mfcc[1:, :].T
        return frame_mfcc
    
    elif feature == 'cens':
        frame_cens = librosa.feature.chroma_cens(y=audio_vector, sr=sampling_rate, hop_length = overlap_length)
        return frame_cens.T

def simple_fast_df(original_path, cover_path):
    
    emb_list = os.listdir(cover_path)
    emb_list = [x for x in emb_list if 'ipynb' not in x]
    
    cens_covers_feature = []
    cens_cvr_name = []
    
    for cvr in emb_list:
        abs_cvr_path = cover_path + cvr
        f = gzip.GzipFile(abs_cvr_path, "r")
        cvr_cens = np.load(f).T
        f.close()
        cens_covers_feature.append(cvr_cens)
        cens_cvr_name.append(cvr)
        
    covers_feature_cens = np.vstack(cens_covers_feature)
    n_sample = len(cens_cvr_name)
    n_timestep = cvr_cens.shape[0]
    n_feat = cvr_cens.shape[1]
    covers_feature_cens = covers_feature_cens.reshape(n_sample, n_timestep, n_feat)
    print(covers_feature_cens.shape)
    
    orig_list = os.listdir(original_path)
    orig_list = [x for x in orig_list if 'ipynb' not in x]
    df_list = []
    
    for orig in tqdm(orig_list):
        f = gzip.GzipFile(original_path + orig, "r")
        feat_cens = np.load(f).T
        f.close()
        # feat_cens = from_audio_to_features(original_path + orig, 'cens')
        distance_seq_cens = []
        chunks = [covers_feature_cens[x:x+500] for x in range(0, len(covers_feature_cens), 500)]
        for chu in chunks:
            mpr, _ = batch_simple_fast_gpu(feat_cens, chu, 30)
            mpr = cp.asnumpy(mpr)
            # mpr = multi_simple_fast(feat_cens, chu, 30)
            distance_seq_tmp = np.median(mpr, axis = 1)
            distance_seq_tmp = distance_seq_tmp.tolist()
            distance_seq_cens.extend(distance_seq_tmp)

        res_df_cens = pd.DataFrame()
        res_df_cens['vid'] = cens_cvr_name
        res_df_cens['distance'] = distance_seq_cens
        res_df_cens['ranked'] = res_df_cens['distance'].rank()
        rank_dict_cens = {x:y for x,y in zip(res_df_cens['vid'], res_df_cens['ranked'])}
        res_df_cens = res_df_cens.sort_values(by = ['distance']).reset_index(drop  = True)
        df_list.append({'original':orig, 'retrieved':res_df_cens})
    return df_list


def query_by_humming_df(original_path, cover_path):
    pass

def rhythm_pattern_df(orignal_path, cover_path):
    
    def eucledian_distance(feature_space, query_vector):
        distance_sum = np.sum((feature_space - query_vector)**2, axis=1)
        return np.sqrt(list(distance_sum))

    def scaled_eucledian_distance(feature_space, query_vector, featureset_lengths):

        distances = (feature_space - query_vector)**2

        # feature_start_idx
        start_idx = 0 

        # normalize distances
        for sequence_length in featureset_lengths:

            # feature_stop_idx
            stop_idx                         = start_idx + sequence_length
            distances[:,start_idx:stop_idx] /= sequence_length#distances[:,start_idx:stop_idx].sum(axis=1).max()
            start_idx                        = stop_idx
        distance_sum = np.sum(distances, axis=1)
        return np.sqrt(list(distance_sum))

    def weighted_eucledian_distance(feature_space, query_vector, featureset_lengths, featureset_weights):

        distances = (feature_space - query_vector)**2

        # feature_start_idx
        start_idx = 0 

        # normalize distances
        for sequence_length, weight in zip(featureset_lengths, featureset_weights):

            # feature_stop_idx
            stop_idx                         = start_idx + sequence_length
            distances[:,start_idx:stop_idx] /= sequence_length#distances[:,start_idx:stop_idx].sum(axis=1).max()
            distances[:,start_idx:stop_idx] *= weight
            start_idx                        = stop_idx
        distance_sum = np.sum(distances, axis=1)
        return np.sqrt(list(distance_sum))

    emb_list = os.listdir(cover_path)
    emb_list = [x for x in emb_list if 'ipynb' not in x]
    
    feat_audios_pop = []
    files = []
    
    for cvr in emb_list:
        abs_cvr_path = cover_path + cvr
        f = open(abs_cvr_path, "rb")
        cvr_cens = pickle.load(f)
        f.close()
        feat_audios_pop.append(cvr_cens)
        files.append(cvr)
        
    vstacked_feat = []
    for i in range(len(feat_audios_pop)):
        concated_feats = []
        feat_name = []
        for key in list(feat_audios_pop[i].keys()):
            feat_shape = feat_audios_pop[i][key].shape[0]
            tmp_feat_name = ['{}_{}'.format(key, x) for x in range(feat_shape)]
            feat_name.extend(tmp_feat_name)
            concated_feats.append(feat_audios_pop[i][key])
        concated_feats = np.concatenate(concated_feats)
        vstacked_feat.append(concated_feats)
    vstacked_feat = np.vstack(vstacked_feat)
    df_rhythm_feat_pop = pd.DataFrame(vstacked_feat, columns=feat_name)
    df_rhythm_feat_pop['title'] = files
    
    feature_data = df_rhythm_feat_pop.iloc[:,:3684].values
    featureset_lengths = [1440, 168, 60, 420, 1176, 420]
    featureset_weights = [0.2, 0.2, 0.3, 0,0.3,0]
    
    orig_list = os.listdir(original_path)
    orig_list = [x for x in orig_list if 'ipynb' not in x]
    df_list = []
    
    for orig in tqdm(orig_list):
        abs_cvr_path = original_path + orig
        f = open(abs_cvr_path, "rb")
        feature_rp = pickle.load(f)
        f.close()
        concated_feats = []
        for key in list(feature_rp.keys()):
            concated_feats.append(feature_rp[key])
        concated_feats = np.concatenate(concated_feats)
        print(concated_feats.shape)
        
        dist = eucledian_distance(feature_data, concated_feats)
        scaled_dist = scaled_eucledian_distance(feature_data, concated_feats, featureset_lengths)
        weighted_dist = weighted_eucledian_distance(feature_data, concated_feats, featureset_lengths, featureset_weights)
        
        display_df = df_rhythm_feat_pop.copy()
        display_df['nothing_distance'] = dist
        display_df['scaled_distance'] = scaled_dist
        display_df['weighted_distance'] = weighted_dist

        display_df['rank_nothing_distance'] = display_df['nothing_distance'].rank(ascending = True)
        display_df['rank_scaled_distance'] = display_df['scaled_distance'].rank(ascending = True)
        display_df['rank_weighted_distance'] = display_df['weighted_distance'].rank(ascending = True)
        display_df = display_df.iloc[:, 3684:]
        df_list.append({'original':orig, 'retrieved':display_df})

    return df_list

def hp_implementation_df(original_path, cover_path):
    pass

def chroma_binary_df(original_path, cover_path):
    def compute_cross_similarity(query_hpcp, cover_hpcp):
        print("TEST 1")
        crp = estd.ChromaCrossSimilarity(
            frameStackSize=9, frameStackStride=1, binarizePercentile=0.095, oti=True
        )
        print("TEST 2")
        pair_crp = crp(query_hpcp, cover_hpcp)
        print("TEST 3")
        score_matrix, distance = estd.CoverSongSimilarity(
            disOnset=0.5,
            disExtension=0.5,
            alignmentType="serra09",
            distanceType="asymmetric",
        )(pair_crp)
        print("TEST 4")
        return distance
    
    def get_rescaled_hpcp_distance(distance):
        lambda_threshold = -math.log(0.88) / 0.161
        score = math.exp(-lambda_threshold * distance)
        print('distance')
        return score
    
    orig_list = os.listdir(original_path)
    emb_list = os.listdir(cover_path)
    # print(emb_list)
    
    df_list = []
    
    for orig in tqdm(orig_list):
        f = gzip.GzipFile(original_path + orig, "r")
        feature = np.load(f).T
        f.close()
        # orig_path = original_path + orig
        # f = open(orig_path, "rb")
        # feature = pickle.load(f)
        # f.close()
        cover_name_list = []
        distances = []
        count_ss = 0
        for cvr in emb_list:
            count_ss = count_ss + 1
            print(count_ss)
            import time
            time.sleep(0.25)
            f = gzip.GzipFile(cover_path + cvr, "r")
            cover_feature = np.load(f)
            f.close()
            # cvr_path = cover_path + cvr
            # f = open(cvr_path, "rb")
            # cover_feature = pickle.load(f)
            # f.close()
            cover_name_list.append(cvr)
            print(feature.shape)
            print(cover_feature.shape)
            distance = get_rescaled_hpcp_distance(compute_cross_similarity(feature, cover_feature))
            distances.append(distance)
            
        df = pd.DataFrame()
        df['vid'] = cover_name_list
        df['distance'] = distances
        df['ranked'] = df['distance'].rank(ascending = False)
        df = df.sort_values(by = ['ranked']).reset_index(drop  = True)
        df_list.append({'original':orig, 'retrieved':df})
        
    return df_list
            
            
        
    



if __name__ == "__main__":
    import pickle
    original_path = "/home/jupyter/rayhan_workdir/personal/perceptual-music-similarity-rayhan-ka/data/covers80/original/hpcp/"
    cover_path = "/home/jupyter/rayhan_workdir/personal/perceptual-music-similarity-rayhan-ka/data/covers80/cover/hpcp/"
    df_list = chroma_binary_df(original_path, cover_path)
    
    with open('./result/chroma_binary_production_dataset.pickle', 'wb') as f:
        pickle.dump(df_list, f)
