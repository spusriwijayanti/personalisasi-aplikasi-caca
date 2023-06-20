import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import numpy as np
import cv2

def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)

def create_pairs_on_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')
    return pairs, y

def initialize_base_network():
    input = Input(shape=(56, 56, 3), name="base_input")
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten(name="flatten_input")(x)
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)

    return Model(inputs=input, outputs=x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def upload_image(key):
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg"], key=key)
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        return image
    return None

def show_image(image):
    st.image(image, use_column_width=True)

def preprocess_image(image):
    resized_image = cv2.resize(image, (56, 56)) / 255.0
    return resized_image

def verify_images(image1, image2, model):
    preprocessed_image1 = preprocess_image(image1)
    preprocessed_image2 = preprocess_image(image2)
    preprocessed_image1 = np.expand_dims(preprocessed_image1, axis=0)
    preprocessed_image2 = np.expand_dims(preprocessed_image2, axis=0)
    distance = model.predict([preprocessed_image1, preprocessed_image2])[0][0]
    return distance

def snn():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    base_network = initialize_base_network()

    input_a = Input(shape=(56, 56, 3))
    input_b = Input(shape=(56, 56, 3))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    try:
        model.load_weights('SNN.h5')
    except:
        st.error("Gagal memuat model SNN")

    if model is not None:
        st.title("Verifikasi atau pencocokan e-KTP dengan foto selfie")
        st.write("Unggah dua foto untuk diverifikasi atau dicocokan.")

        st.subheader("Unggah gambar e-KTP")
        image1 = upload_image(key="image1")
        if image1 is not None:
            show_image(image1)

        st.subheader("Unggah foto selfie")
        image2 = upload_image(key="image2")
        if image2 is not None:
            show_image(image2)

        if st.button("Verifikasi"):
            if image1 is None or image2 is None:
                st.warning("Silakan unggah kedua foto")
            else:
                similarity_score = verify_images(image1, image2, model)
                st.write(f"Skor atau nilai kemiripan: {similarity_score}")

snn()
