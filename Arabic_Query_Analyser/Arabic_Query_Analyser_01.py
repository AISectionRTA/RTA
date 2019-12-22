import gensim
import os
import argparse
from utilities import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

path = os.path.join(os.path.join(os.getcwd(), 'Data'), 'Word2Vec')
w2v_path = os.path.join(path, 'full_grams_sg_300_twitter.mdl')
t_model = gensim.models.Word2Vec.load(w2v_path)
lexicondpath = os.path.join(path, 'Lexicon.txt')
lexicon = []

with open(lexicondpath, mode='r', encoding='utf-8-sig') as sw:
    for cnt, line in enumerate(sw):
        lexicon.append(clean_str(str(line).split()[0]))


def cosin(a, b):
    return cosine_similarity(np.asanyarray([a]), np.asanyarray([b]), dense_output=True)[0][0]


def most_Similar(word='برجر', listofword=['طعام', 'سيارة', 'شراب']):
    word = clean_str(word)
    for i, w in enumerate(listofword):
        listofword[i] = clean_str(w)
    if word in t_model.wv:
        max = 0.0
        max_word = ''
        for term in listofword:
            if term in t_model.wv:
                c = cosin(t_model[word], t_model.wv.__getitem__(term))
                if max <= c:
                    max = c
                    max_word = term
        return max_word
    else:
        return ''


def getmostsimilar(tokensList):
    for token in set(tokensList):
        if token in t_model.wv:
            print(token)
            most_similar = t_model.wv.most_similar(token, topn=10)
            for term, score in most_similar:
                term = clean_str(term).replace(" ", "_")
                if term != token:
                    print(term, score)


def enhance_query(tokensList):
    query = []
    for token in tokensList:
        if token in lexicon:
            query.append(token)
        else:
            query.append(most_Similar(token, lexicon))
    return ' '.join(query)


def main():
    text = u'كيف نسير دبي'
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--text", required=False,
                    help="Enter Text")
    args = vars(ap.parse_args())
    if args["text"] is not None:
        text = args["text"]
    tokens = clean_str(text).split()  # .replace(" ", "_")
    tokensList = [token for token in set(tokens)]
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            tokensList.append(tokens[i] + '_' + tokens[j])
            tokensList.append(tokens[j] + '_' + tokens[i])
    print(enhance_query(tokensList))


if __name__ == '__main__':
    main()
