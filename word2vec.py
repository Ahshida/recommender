from gensim.models import Word2Vec

filename = 'GoogleNews-vectors-negative300.bin.gz'
word2vec_model = Word2Vec.load_word2vec_format(filename, binary=True)

word2vec_model.init_sims(replace=True)

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)

def has_vector_representation(word2vec_model, doc):

    return not all(word not in word2vec_model.vocab for word in doc)

corpus, texts, y = filter_docs(corpus, texts, y,
                               lambda doc: has_vector_representation(word2vec_model, doc))

snippets, _, snippets_labels = filter_docs(snippets, None, snippets_labels,
                                           lambda doc: has_vector_representation(word2vec_model, doc))

from sklearn.metrics.pairwise import cosine_similarity

sims['ng20']['centroid'] = cosine_similarity(np.array([document_vector(word2vec_model, doc)
                                                       for doc in corpus]))

sims['snippets']['centroid'] = cosine_similarity(np.array([document_vector(word2vec_model, doc)
                                                           for doc in snippets]))

