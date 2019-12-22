from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import os
import numpy as np
import tensorflow as tf
import argparse
import re

stopWords = []
path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data'), 'Risk Analysis')
# print('\n\n\n\n'+path+'\n\n\n\n\n')
model = tf.keras.models.load_model(os.path.join(path, 'Risk_Level'))
model.load_weights(os.path.join(path, 'Risk_Level'))
encoder = tfds.features.text.TokenTextEncoder.load_from_file(os.path.join(path, 'encoder'))


# model.summary()

def removeStopWord(text):
    word_tokens = text.split()
    filtered_sentence = [w for w in word_tokens if not w in stopWords]
    return {" ".join(filtered_sentence)}


def clean(text):
    text = str(removeStopWord(text)).lower()
    # getVals = list([num2words(val) if val.isnumeric() else val
    #               for val in text if val.isalpha() or val.isnumeric() or val is ' '])
    getVals = list([val for val in text if val.isalpha() or val is ' '])
    result = "".join(getVals)
    return re.sub(' +', ' ', result)


def getLevel_softmax(prid):
    pes = ['Low', 'Medium', 'High']
    print(prid)
    return pes[np.argmax(prid)]


def sample_predict(sentence):
    encoded_sample_pred_text = encoder.encode(sentence)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--text", required=False,
                    help="Enter Text")
    args = vars(ap.parse_args())
    if args["text"] != None:
        sample_pred_text = args["text"]
    else:
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
        """
                sample_pred_text = 'Corporate Administrative Support Services Sector ' \
                                   'Administration Services	Classification of RTA Documents, Standardization of Retention Periods ' \
                                   'Monday, 30 September, 2019	escalate this issue to respective team in each agency/ sector ' \
                                   'Resistance from agencies sectors to comply with project requirements ' \
                                   'Resistance from agencies sectors to comply with project requirements'
        """
        print('Please enter a text to classify')

    predictions = sample_predict(clean(str(sample_pred_text)))
    # print(sample_pred_text)
    # print(getLevel_softmax(predictions))
    print(getLevel_softmax(predictions))


if __name__ == '__main__':
    main()
