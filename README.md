# perceptual-music-similarity-rayhan-ka
This is a project to retrieve N similar musics from an Input

## 1. How to Derive Features

To derive features for this project, follow these steps:

1. Run the feature extraction script: efficient_features.py
2. Uncomment frame_cens / frame_hpcp

## 2. How to Run Inference After Features

Once you have derived the features, you can run inference using the following steps:

1. Run inference_logic.py
2. In the main part, select function to run. Make sure the path is already exist.

## 3. How to Evaluate Inference

See check_result.ipynb to get metrics for inference

## 4. Where to Look for Complete Data

The complete dataset used in this project can be found in my drive (soon)

## 5. How to run the prototype

To run the prototype you need to run streamlit run prototype_demo.py. Create song_db folder first.
Once it ran, it will have two modes: Update & Compare. Update meaning adding songs to song_db, processing uploaded song's features.
Compare meaning given a song, it returns most similar songs.
