from sklearn.metrics.pairwise import pairwise_distances

A = np.array([[i] for i in range(len(snippets))])

def f(x, y):
    return word2vec_model.wmdistance(snippets[int(x)], snippets[int(y)])

X_wmd_distance_snippets = pairwise_distances(A, metric=f, n_jobs=-1)

def most_similar(i, X_sims, topn=None):
    """return the indices of the topn most similar documents with document i
    given the similarity matrix X_sims"""

    r = np.argsort(X_sims[i])[::-1]
    if r is None:
        return r
    else:
        return r[:topn]

#LSI
print(most_similar(0, sims['ng20']['LSI'], 20))
print(most_similar(0, sims['snippets']['LSI'], 20))

#Centroid
print(most_similar(0, sims['ng20']['centroid'], 20))
print(most_similar(0, sims['snippets']['centroid'], 20))

from gensim.similarities import WmdSimilarity

wmd_similarity_top20 = WmdSimilarity(corpus, word2vec_model, num_best=20)
most_similars_wmd_ng20_top20 = wmd_similarity_top20[corpus[0]]

wmd_similarity_snippets = WmdSimilarity(snippets, word2vec_model, num_best=20)
most_similars_snippets = wmd_similarity_snippets[snippets[0]]