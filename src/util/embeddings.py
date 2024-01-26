import os.path

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.preprocessing import normalize

nltk.download('punkt')
model = Word2Vec.load(os.path.join(os.path.dirname(__file__), 'w2v_model.bin'))


def calculate_embedding(text):
    # Токенизация текста
    tokens = word_tokenize(text)
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]

    if word_vectors:
        average_embedding = normalize([np.mean(word_vectors, axis=0)])
        return average_embedding
    else:
        return None
