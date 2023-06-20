import streamlit as st
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
import pandas as pd
from keras.engine.sequential import Sequential as KerasSequential

@st.cache(hash_funcs={KerasSequential: lambda _: None})
def load_model(weights_path):

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False

    flatten_layer = Flatten()
    dense_layer_1 = Dense(64, activation='relu')
    dropout_layer = Dropout(0.5)
    prediction_layer = Dense(1, activation='sigmoid')

    model = Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dropout_layer,
        prediction_layer
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_path)
    return model

def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))/255
    return resized_image

def recommend_cafe(kategori, hasil_pred):
    cafes = []
    for i in range(0, len(hasil_pred), 4):
        tigajamlebih = kategori[kategori[:, 1] == hasil_pred[i]]
        limapuluhribulebih = kategori[kategori[:, 2] == hasil_pred[i + 1]]
        makan_minum = kategori[kategori[:, 3] == hasil_pred[i + 2]]
        malamhari = kategori[kategori[:, 4] == hasil_pred[i + 3]]

        cafes_tigajamlebih = tigajamlebih[tigajamlebih[:, 6] == 'k_wifi'][:, 0]
        cafes_limapuluhribulebih = limapuluhribulebih[limapuluhribulebih[:, 7] == 'k_mahal'][:, 0]
        cafes_makan_minum = makan_minum[makan_minum[:, 9] == 'k_mkn_mnm'][:, 0]
        cafes_malamhari = malamhari[malamhari[:, 10] == 'k_24jam'][:, 0]

        recommended_cafes = np.intersect1d(np.intersect1d(np.intersect1d(cafes_tigajamlebih, cafes_limapuluhribulebih), cafes_makan_minum), cafes_malamhari)
        if len(recommended_cafes) > 0:
            cafes.append(recommended_cafes)
        else:
            cafes.append("Tidak ada rekomendasi cafe")

    return cafes

def load_data():
    kategori = pd.read_csv('kategori3.csv', delimiter=';', encoding='latin-1')
    kategori = kategori.astype(str)
    kategori = np.array(kategori)
    kategori = kategori.astype('<U55')

    kuesioner = pd.read_csv('kuesioner_hasilprediksi2.csv', delimiter=';')
    kuesioner = np.array(kuesioner)
    kuesioner = kuesioner.astype(str)

    return kategori, kuesioner

def predict_cafe(kategori, kuesioner, nama_pengguna):
    # Cari baris yang sesuai dengan nama pengguna
    pengguna_row = None
    for row in kuesioner:
        if row[0] == nama_pengguna:
            pengguna_row = row
            break
    
    if pengguna_row is None:
        return None  # Jika nama pengguna tidak ditemukan
    
    hasil_pred = pengguna_row[1:]

    results = []  # List untuk menyimpan hasil prediksi
    for i in range(0, len(hasil_pred), 4):
        tigajamlebih = kategori[kategori[:, 1] == hasil_pred[i]]
        limapuluhribulebih = kategori[kategori[:, 2] == hasil_pred[i + 1]]
        makan_minum = kategori[kategori[:, 3] == hasil_pred[i + 2]]
        malamhari = kategori[kategori[:, 4] == hasil_pred[i + 3]]

        b = np.intersect1d(tigajamlebih[:, 0], limapuluhribulebih[:, 0])
        c = np.intersect1d(makan_minum[:, 0], b)
        d = np.intersect1d(malamhari[:, 0], c)
        results.append(d)

    return results

def fungsi_dua():
    st.title("Rekomendasi cafe")
    st.write("Unggah foto selfie anda untuk dapatkan rekomendasi cafe!")

    uploaded_file = st.file_uploader("Pilih foto selfie", type=['jpg'])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        processed_image = preprocess_image(image)

        model_paths = [
            'vgg163jamlebihdropout.h5',
            'vgg1650ribulebihdropout.h5',
            'vgg16asamdropout.h5',
            'vgg16begadangdropout.h5',
            'vgg16ekstrovertdropout.h5',
            'vgg16kelamindropout.h5',
            'vgg16keramaiandropout.h5',
            'vgg16makanminumdropout.h5',
            'vgg16malamharidropout.h5',
            'vgg16manisdropout.h5',
            'vgg16pahitdropout.h5'
        ]
        models = []
        labels = ['3jamlebih', '50ribulebih', 'asam', 'begadang', 'ekstrovert', 'kelamin', 'keramaian', 'makan_minum', 'malamhari', 'manis', 'pahit']
        for model_path in model_paths:
            model = load_model(model_path)
            models.append(model)

        predictions = []
        for model in models:
            prediction = model.predict(np.expand_dims(processed_image, axis=0))
            predictions.append(prediction[0][0])

        st.write("Hasil Prediksi Kesukaan:")
        for i, label in enumerate(labels):
            if label == 'kelamin':
                if predictions[i] > 0.5:
                    st.write(label + ": Perempuan")
                else:
                    st.write(label + ": Laki-Laki")
            else:
                if predictions[i] > 0.5:
                    st.write(label + ": Suka")
                else:
                    st.write(label + ": Tidak Suka")

        kategori, kuesioner = load_data()

        nama_pengguna = uploaded_file.name.split('.')[0]
        
        results = predict_cafe(kategori, kuesioner, nama_pengguna)

        st.write("Rekomendasi Cafe:")
        if results is None:
            st.write("Nama pengguna tidak ditemukan.")
        else:
            for result in results:
                st.write(result)

if __name__ == "__main__":
    fungsi_dua()
