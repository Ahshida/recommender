from sklearn.datasets import fetch_20newsgroups

from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords

download('punkt')
download('stopwords')

stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc

# Fetch ng20 dataset
ng20 = fetch_20newsgroups(subset='train',
                          remove=('headers', 'footers', 'quotes'))

# text and ground truth labels
texts, y = ng20.data, ng20.target

corpus = [preprocess(text) for text in texts]

def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)

corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: (len(doc) != 0))

snippets = []
snippets_labels = []
snippets_file = "data/data-web-snippets/train.txt"
with open(snippets_file, 'r') as f:
    for line in f:
        # each line is a snippet: a bag of words separated by spaces and
        # the category
        line = line.split()
        category = line[-1]
        doc = line[:-1]
        snippets.append(doc)
        snippets_labels.append(category)

snippets, _, snippets_labels = filter_docs(snippets, None, snippets_labels,
                                           lambda doc: (len(doc) != 0))

                                           