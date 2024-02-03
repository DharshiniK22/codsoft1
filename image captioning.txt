# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

# Define constants
MAX_SEQ_LENGTH = 20
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
IMAGE_FEATURE_DIM = 2048

# Load pre-trained ResNet50 model
resnet_model = ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
image_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)

# Define the captioning model
input_img_features = Input(shape=(IMAGE_FEATURE_DIM,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(EMBEDDING_DIM, activation='relu')(inp_img1)

input_captions = Input(shape=(MAX_SEQ_LENGTH,))
inp_cap1 = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

decoder1 = keras.layers.add([inp_img2, inp_cap3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder2)

# Combine image and caption inputs into the model
captioning_model = Model(inputs=[input_img_features, input_captions], outputs=outputs)

# Compile the model
captioning_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to generate captions
def generate_caption(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    img_features = image_model.predict(img).reshape(1, IMAGE_FEATURE_DIM)

    # Start the captioning process
    caption_input = np.zeros((1, MAX_SEQ_LENGTH))

    caption = 'startseq'
    for i in range(MAX_SEQ_LENGTH):
        seq = [word2idx[word] for word in caption.split() if word in word2idx]
        seq = pad_sequences([seq], maxlen=MAX_SEQ_LENGTH)
        prediction = captioning_model.predict([img_features, seq], verbose=0)
        prediction = np.argmax(prediction)
        word = idx2word[prediction]
        if word is None:
            break
        caption += ' ' + word
        if word == 'endseq':
            break
    return caption

# Example usage
image_path = "C:\Users\myhome\Downloads\anime-sunset-piano-clouds-wallpaper-preview.jpg"'
caption = generate_caption(image_path)
print('Generated Caption:', caption)
