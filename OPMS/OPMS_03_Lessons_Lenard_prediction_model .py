from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.backend import manual_variable_initialization
import plotly.graph_objects as go

manual_variable_initialization(True)

path = os.path.join(os.path.join(os.getcwd(), 'Data'), 'Lessons Lenard')
datapath = os.path.join(path, 'Lessons Lenard.csv')
stopwordpath = os.path.join(path, 'stopWords.txt')
stopWords = []
cols = ['Data', 'Category']
raw_Data = []
labels = []
temp = []
labeled_data_sets = []
BUFFER_SIZE = 10
BATCH_SIZE = 20
TAKE_SIZE = 240
train_Test_Data = []

Category = ['Testing', 'Technical', 'Stakeholders & Communication', 'Schedule & Priorities', 'Risk Issues',
            'Resources', 'Requirements', 'Quality QHSE', 'Management', 'HR', 'General', 'Contract', 'Claims & Disputes',
            'Budget Cost', 'Administration', 'all']


def plot_graphs(history, string):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history[string],
        name=string
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_' + string],
        name='val_' + string
    ))
    fig.update_layout(xaxis_title="Epochs", yaxis_title=string)
    fig.show()


with open(stopwordpath, mode='r', encoding='utf-8-sig') as sw:
    for cnt, line in enumerate(sw):
        # print("Line {}: {}".format(cnt, line))
        stopWords.append(str(line).split()[0])


def removeStopWord(text):
    word_tokens = text.split()
    filtered_sentence = [w for w in word_tokens if not w in stopWords]
    return {" ".join(filtered_sentence)}


def clean(text):
    text = str(removeStopWord(text)).lower()
    getVals = list([val for val in text if val.isalpha() or val.isnumeric() or val is ' '])
    result = "".join(getVals)
    return re.sub(' +', ' ', result)


Category2 = [clean(clc) for clc in Category]


def getLable(f):
    return Category2.index(f)


with open(datapath, mode='r', encoding='utf-8-sig') as sw:
    line_count = 0
    for cnt, line in enumerate(sw):
        # print(row)
        temp.clear()
        if line_count == 0:
            pass
        else:
            temp = [clean(val) for i, val in enumerate(str(line).split(','))]
            raw_Data.append(tf.constant(re.sub(' +', ' ', ' '.join([val for val in temp[0:len(temp) - 1]]))))
            labels.append(tf.cast(getLable(temp[len(temp) - 1]), tf.int64))
        line_count += 1

dataset = tf.data.Dataset.from_tensor_slices((raw_Data, labels))
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text_tensor, _ in dataset:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
encoder.save_to_file(os.path.join(path, 'encoder'))

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


encoded_data = dataset.map(encode_map_fn)
train_data = encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

test_data = encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

# train_data, test_data = make_dataset(encoded_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], [])), k_fold)

vocab_size = len(vocabulary_set) + 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
#model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(Category) - 1, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_data, epochs=500, validation_data=test_data)
eval_loss, eval_acc = model.evaluate(test_data)
print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
model.save(os.path.join(path, 'Lessons_Lenard'))
model.save_weights(os.path.join(path, 'Lessons_Lenard'))
