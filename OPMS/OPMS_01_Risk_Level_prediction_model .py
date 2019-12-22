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

k_fold = 10
path = os.path.join(os.path.join(os.getcwd(), 'Data'), 'Risk Analysis')
datapath = os.path.join(path, 'Risk analysis Level2.csv')
stopwordpath = os.path.join(path, 'stopWords.txt')
stopWords = []
cols = ['Data', 'Level']
raw_Data = []
labels = []
temp = []
labeled_data_sets = []
BUFFER_SIZE = 1000
BATCH_SIZE = 50
TAKE_SIZE = 1200
train_Test_Data = []


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


def getLable(f):
    if f == 'low':
        return 0
    elif f == 'medium':
        return 1
    else:
        return 2


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
            labels.append(tf.cast(getLable(clean(temp[len(temp) - 1])), tf.int64))
        line_count += 1

dataset = tf.data.Dataset.from_tensor_slices((raw_Data, labels))

for ex in dataset.take(5):
    print(ex)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in dataset:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

print(vocabulary_set)

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
example_text = next(iter(dataset))[0].numpy()
print(example_text)
encoder.save_to_file(os.path.join(path, 'encoder'))

encoded_example = encoder.encode(example_text)
print(encoded_example)


def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


encoded_data = dataset.map(encode_map_fn)

for ex in encoded_data.take(5):
    print(ex)

train_data = encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

test_data = encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

# train_data, test_data = make_dataset(encoded_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], [])), k_fold)

vocab_size = len(vocabulary_set) + 2


def getLevel_softmax(prid):
    pes = ['Low', 'Medium', 'High']
    print(prid)
    return pes[np.argmax(prid)]


def sample_predict(sentence):
    encoded_sample_pred_text = encoder.encode(sentence)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)


"""
sample_pred_text = 'Corporate Administrative Support Services Sector ' \
                   'Building & Facilities ' \
                   'Oasis of Innovation	Saturday, 10 December, 2016	Follow up ' \
                   'Follow up ' \
                   'Contractors not submmiting their offer	No offers'
"""

sample_pred_text = 'Corporate Administrative Support Services Sector ' \
                   'Administration Services	Accreditation for RTA Nursery based on EYFS Criteria ' \
                   '(Early years foundation stage) – Phase I ' \
                   'Thursday, 31 October, 2019	Keep following up with the proxy accreditation body. ' \
                   'Look for an alternative accreditation body.	Impact on Administration Services Department ' \
                   'Happiness on "Nursery Services". ' \
                   'Delay of Receiving Accreditation'

"""
sample_pred_text = 'Corporate Administrative Support Services Sector ' \
                   'Administration Services	Classification of RTA Documents, Standardization of Retention Periods ' \
                   'Monday, 30 September, 2019	escalate this issue to respective team in each agency/ sector ' \
                   'Resistance from agencies sectors to comply with project requirements ' \
                   'Resistance from agencies sectors to comply with project requirements'
"""

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(3, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64, activation='relu'))

# model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, epochs=3, validation_data=test_data)
eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))

predictions = sample_predict(sample_pred_text)
print(sample_pred_text)
# print(getLevel_softmax(predictions))
print(getLevel_softmax(predictions))

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
print('history.history[string]')
print(history.history['accuracy'])
print('history.history[val_ + accuracy')
print(history.history['val_' + 'accuracy'])

model.save(os.path.join(path, 'Risk_Level'))
model.save_weights(os.path.join(path, 'Risk_Level'))
