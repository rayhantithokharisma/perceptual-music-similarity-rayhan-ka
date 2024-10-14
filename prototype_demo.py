import streamlit as st
import librosa
import numpy as np
import pandas as pd
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

import streamlit as st
import os
import librosa
import numpy as np
import pickle
import math

# Function to process MP3 file and save output
def process_mp3(file):

    with open(file.name, 'wb') as f:
        f.write(file.getbuffer())

    sampling_rate = 16000
    segment_duration = 0.4
    overlap_duration = 0.1

    segment_length = int(segment_duration * sampling_rate)
    overlap_length = int(overlap_duration * sampling_rate)
    audio_vector, sampling_rate_1 = librosa.load(file.name, sr = None)

    if sampling_rate_1 != sampling_rate:
        audio_vector = librosa.resample(audio_vector, orig_sr=sampling_rate_1, target_sr=sampling_rate)
    audio_vector = librosa.util.fix_length(audio_vector, size=sampling_rate * 181)
    audio_vector = audio_vector[:int(sampling_rate * 180)]
    
    frame_hpcp = hpcpgram(audio_vector, 
                    sampleRate=sampling_rate, frameSize=segment_length, hopSize=segment_length - overlap_length)
    
    f = open('./song_db/' + file.name.replace('.mp3', '') +  '.gz', 'wb')
    pickle.dump(frame_hpcp, f)
    f.close()

    os.remove(file.name)
    
    return f"Processed and saved {file.name}"

def compute_cross_similarity(query_hpcp, cover_hpcp):
    crp = estd.ChromaCrossSimilarity(
        frameStackSize=9, frameStackStride=1, binarizePercentile=0.095, oti=True
    )
    pair_crp = crp(query_hpcp, cover_hpcp)
    score_matrix, distance = estd.CoverSongSimilarity(
        disOnset=0.5,
        disExtension=0.5,
        alignmentType="serra09",
        distanceType="asymmetric",
    )(pair_crp)
    return distance

def get_rescaled_hpcp_distance(distance):
    lambda_threshold = -math.log(0.88) / 0.161
    score = math.exp(-lambda_threshold * distance)
    return score

# Function to compare MP3 file with saved features
def compare_mp3(file):
    with open(file.name, 'wb') as f:
        f.write(file.getbuffer())

    sampling_rate = 16000
    segment_duration = 0.4
    overlap_duration = 0.1

    segment_length = int(segment_duration * sampling_rate)
    overlap_length = int(overlap_duration * sampling_rate)
    audio_vector, sampling_rate_1 = librosa.load(file.name, sr = None)

    if sampling_rate_1 != sampling_rate:
        audio_vector = librosa.resample(audio_vector, orig_sr=sampling_rate_1, target_sr=sampling_rate)
    audio_vector = librosa.util.fix_length(audio_vector, size=sampling_rate * 181)
    audio_vector = audio_vector[:int(sampling_rate * 180)]
    
    frame_hpcp = hpcpgram(audio_vector, 
                    sampleRate=sampling_rate, frameSize=segment_length, hopSize=segment_length - overlap_length)
    
    emb_list = os.listdir('./song_db/')
    cover_name_list = []
    distances = []
    for cvr in emb_list:
        cvr_path = './song_db/' + cvr
        f = open(cvr_path, "rb")
        cover_feature = pickle.load(f)
        f.close()
        cover_name_list.append(cvr)
        
        distance = get_rescaled_hpcp_distance(compute_cross_similarity(frame_hpcp, cover_feature))
        distances.append(distance)
        
    df = pd.DataFrame()
    df['vid'] = cover_name_list
    df['distance'] = distances
    df['ranked'] = df['distance'].rank(ascending = False)
    df = df.sort_values(by = ['ranked']).reset_index(drop  = True)
    os.remove(file.name)
    
    return df.iloc[:10]

# Streamlit app
def main():
    st.title("Music Processing App")
    st.write(f"Current number of songs in the database: {len(os.listdir('./song_db/'))}")

    mode = st.sidebar.selectbox("Choose Mode", ["Update DB", "Compare"])
    
    if mode == "Update DB":
        st.header("Update Database")
        uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")
        
        if uploaded_file is not None:
            result = process_mp3(uploaded_file)
            st.success(result)
    
    elif mode == "Compare":
        st.header("Compare with Database")
        uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")
        
        if uploaded_file is not None:
            results = compare_mp3(uploaded_file)
            st.dataframe(results)

if _name_ == "_main_":
    main()