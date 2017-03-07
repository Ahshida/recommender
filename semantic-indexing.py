import numpy as np
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity

sims = {'ng20': {}, 'snippets': {}}
dictionary = corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sims['ng20']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                for i in range(len(corpus))])


dictionary_snippets = corpora.Dictionary(snippets)
corpus_gensim_snippets = [dictionary_snippets.doc2bow(doc) for doc in snippets]
tfidf_snippets = TfidfModel(corpus_gensim_snippets)
corpus_tfidf_snippets = tfidf_snippets[corpus_gensim_snippets]
lsi_snippets = LsiModel(corpus_tfidf_snippets,
                        id2word=dictionary_snippets, num_topics=200)
lsi_index_snippets = MatrixSimilarity(lsi_snippets[corpus_tfidf_snippets])
sims['snippets']['LSI'] = np.array([lsi_index_snippets[lsi_snippets[corpus_tfidf_snippets[i]]]
                                    for i in range(len(snippets))])

