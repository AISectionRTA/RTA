import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

path = os.path.join(os.path.join(os.getcwd(), 'Data'), 'Word2Vec')
w2v_path = os.path.join(path, 'GoogleNews-vectors-negative300.bin')
w2v_model = KeyedVectors.load_word2vec_format(datapath(w2v_path), binary=True)
datapath = os.path.join(
    os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data'), 'Lessons Lenard'),
    'Lessons Lenard.csv')
size = 25  # size of chart (number of words to display)
level = ['Testing', 'Technical', 'Stakeholders_&_Communication', 'Schedule_&_Priorities', 'Risk_Issues', 'Resources',
         'Requirements', 'Quality_QHSE', 'Management', 'HR', 'General', 'Contract', 'Claims & Disputes',
         'Budget_Cost', 'Administration', 'all']
l = 15
stopWords = []
data = []
temp = []
cols = []
bagOfWords = []
corpus = []
classIndex = 7
getter = [0, 1, 2, 3, 4, 5, 6, classIndex]

stopwordpath = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data'), 'stopWords.txt')

with open('Data/stopWords.txt', mode='r', encoding='utf-8-sig') as sw:
    for cnt, line in enumerate(sw):
        # print("Line {}: {}".format(cnt, line))
        stopWords.append(str(line).split()[0])


def removeStopWord(text):
    word_tokens = text.split()
    filtered_sentence = [w for w in word_tokens if not w in stopWords]
    return {" ".join(filtered_sentence)}


def clean(text):
    text = str(removeStopWord(text)).lower()
    getVals = list([val for val in text if val.isalpha() or val is ' '])
    result = "".join(getVals)
    return re.sub(' +', ' ', result)


def tsne_bubble(model, lexicon):
    labels = []
    tokens = []
    x = []
    y = []
    marker_size = []
    for word in lexicon[0:size]:
        if word[0] in model.wv.vocab:
            tokens.append(model[word[0]])
            labels.append(word[0])
            marker_size.append(word[1] * 5000)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    fig = go.Figure(data=[go.Scatter(
        x=x, y=y,
        mode='markers+text',
        text=labels,
        marker_size=marker_size)
    ])

    fig.show()


def tsne_plot2d(words_List):
    fig = go.Figure(
        data=[go.Bar(
            x=list(x[0] for x in words_List)[0:size],
            y=list(x[1] for x in words_List)[0:size])
        ], layout_title_text="TFIDF For " + level[l] + " levels",
    )
    fig.update_layout(
        xaxis_title="TFIDF Value for " + level[l] + " levels",
        yaxis_title="Terms",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()


def tsne_plot3d(model, lexicon):
    labels = []
    tokens = []
    X = []
    Y = []
    Z = []
    dictList = list()

    for word in lexicon[0:size]:
        if word[0] in model.wv.vocab:
            tokens.append(model[word[0]])
            labels.append(word[0])

    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    for value in new_values:
        X.append(value[0])
        Y.append(value[1])
        Z.append(value[2])
    for x, y, z, labl in zip(X, Y, Z, labels):
        dictList.append(
            dict(showarrow=False, x=x, y=y, z=z, text=labl, xanchor="left", xshift=5)
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="markers", name="z", ))
    fig.update_layout(
        scene=go.layout.Scene
            (
            # aspectratio=dict(x=1, y=1, z=1),
            camera=dict(center=dict(x=0, y=0, z=0), eye=dict(x=1, y=1, z=1), up=dict(x=2, y=2, z=2)),
            dragmode="turntable",
            xaxis=dict(title_text="", type="linear"),
            yaxis=dict(title_text="", type="linear"),
            zaxis=dict(title_text="", type="linear"),
            annotations=dictList
        ), xaxis=dict(title_text="x"), yaxis=dict(title_text="y"))
    fig.show()


with open(datapath, mode='r', encoding='utf-8-sig') as sw:
    line_count = 0
    for cnt, line in enumerate(sw):
        # print(row)
        temp.clear()
        if line_count == 0:
            cols = list(['cols' + str(colscoun + 1) if val is '' else val
                         for val, colscoun in zip(str(line).split(','), range(len(str(line).split(','))))])
            bagOfWords.append(w for w in cols if len(w) > 0)
        else:
            # temp = [clean(val) for val in str(line).split(',')]
            temp = [clean(val) for i, val in enumerate(str(line).split(',')) if i in getter]
            if len(temp) > 1:
                if temp[-1] == level[l] or level[l] == 'all':
                    print(temp[-1])
                    corpus.append(re.sub(' +', ' ', ' '.join([val for val in temp[0:-1]])))
                    data.append(temp.copy())
                    bagOfWords.append(w for w in temp[0:-1] if len(w) > 0)
                else:
                    print(temp[-1])
        line_count += 1

uniqueWords = set(bagOfWords)
numOfWords = dict.fromkeys(uniqueWords, 0)
for word in bagOfWords:
    numOfWords[word] += 1

newcols = [cols[i] for i in getter]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)

np_tfidf = np.average(np.array(tfidf.toarray()), axis=0)

tfidf_dic = {}
for key, val in zip(vectorizer.get_feature_names(), np_tfidf):
    tfidf_dic[key] = val

sorted_tfidf = sorted(tfidf_dic.items(), key=lambda kv: kv[1], reverse=True)

sorted_tf = sorted(numOfWords.items(), key=lambda kv: kv[1], reverse=True)

print('TFIDF                   Word        ')
for t, itm in zip(list(x[0] for x in sorted_tfidf)[0:size], list(x[1] for x in sorted_tfidf)[0:size]):
    print(str(itm) + '     ' + str(t))

tsne_plot2d(sorted_tfidf)
tsne_plot3d(w2v_model, sorted_tfidf)
tsne_bubble(w2v_model, sorted_tfidf)
