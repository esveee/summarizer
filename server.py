from flask import Flask, request, render_template

import json
import numpy as np
import pandas as pd
import nltk
import networkx
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import jinja2

jinja_environment = jinja2.Environment(autoescape=True,loader=jinja2.FileSystemLoader('templates'))


nltk.download('punkt')                                                                                                                      # one time execution
nltk.download('stopwords')
app = Flask(__name__)

@app.route('/')
def static_page():
    return render_template('index.html')

def script():                                                                                                                                           # single domain multiple documentation article
                                                                                                                                            # single domain multiple documentation article
    df = pd.read_csv(r'C:\Users\samanvayvajpayee\Downloads\tennis_articles_v4.csv', encoding='utf-8')
    sentences = []
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))

    sentences = [y for x in sentences for y in x]

    # Extract word vectors
    # GloVe- word embeddings are vector representation of words.
    # using GloVe also for maintaining the order
    word_embeddings = {}
    # f = open(r'Desktop\textrank\glove.6B.100d.txt', encoding='utf-8')                        #Download glove.6B.100d.txt embedding and replace the file address accordingly
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        word_embeddings[word] = coefs
    f.close()

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]


    # function to remove stopwords
    def remove_stop_words(sen, lang='English'):
        stop_words = stopwords.words(lang)
        sentence_new = " ".join([i for i in sen if i not in stop_words])
        return sentence_new


    # remove stopwords
    clean_sentences = [remove_stop_words(r.split()) for r in clean_sentences]

    # create a word-vector each with size 100 of each sentence
    sentence_vectors = []
    for sen in clean_sentences:
        if len(sen) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in sen.split()])/(len(sen.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    # cosine similarity to check similarity between sentences
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100),
                                                  sentence_vectors[j].reshape(1, 100))[0, 0]

    # making a graph by applying pageRank algo
    nx_graph = networkx.from_numpy_array(sim_mat)
    scores = networkx.pagerank(nx_graph)
    ranked_scores = sorted(((scores[i], s) for i,s in enumerate(sentences)), reverse=True)

    # Extract top 2 sentences as the summary
    for i in range(3):
        s+=(ranked_scores[i][1])
    return s


    summ=""
@app.route("/script", methods=['GET','POST'])

def summarize():
    #if request.method == 'GET':
    #     input_string = request.form['text']

    #if request.method == 'POST':
    #    request.form['sum']
        summ = script()
        return render_template('index.html', summary=summ)


if __name__ == "__main__":
    app.run()
