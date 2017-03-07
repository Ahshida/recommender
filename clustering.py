from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi

sims['snippets']['WMD'] = -X_wmd_distance_snippets + X_wmd_distance_snippets.max() - 1

for dataset in ['ng20', 'snippets']:
    for method in ['centroid', 'LSI', 'WMD']:
        if not (method == 'WMD' and dataset == 'ng20'):
            if dataset == 'ng20':
                n_clusters = 20
            else:
                n_clusters = 8

            sc = SpectralClustering(n_clusters=n_clusters,
                                    affinity='precomputed')

            matrix = sims[dataset][method]
            matrix += 1

            sc.fit(matrix)

            labels = None
            if dataset == 'ng20':
                labels = y
            else:
                labels = snippets_labels

            print("Method: {}\nDataset: {}".format(method, dataset))
            print(nmi(sc.labels_, labels))
            print("-"*10)