######## PRINCIPAL COMPONENT ANALYSIS ##########
# By Felix Burmester, Alexander Valentini and Anton Jørgensen
################################################



################################################
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.linalg import eigh
from text_classifier_main_final import TextClassifier
################################################

################################################
# Enable/disable by saying True/False
# Pick only one True per pair!
Baseline = True
fastText = False

# News or Spam data to be plotted
News = False
Spam = True

# Training or test data to be plotted
Train_data_pca = False
Test_data_pca = True

# Input a title for plot (fx. fastText News Test 15 dim)
Title = "Baseline Spam Test 50 dim"

# Choose embedded vectors' dimension for fastText
if fastText:
    embedded_dim = 45
    # Choose number of epochs
    epo = 5
    # Choose number of ngrams
    n_grams = 3
    
# Choose to remove stopwords from Baseline method (deafult = False)
if Baseline:
    Stopwords = False
    stopwords = [".",",",":",";","...","-"]
################################################

################################################
# Load data
if News:
    news_data = np.load ("./news_data.npz", allow_pickle = True )
    train_texts = news_data ["train_texts"]
    test_texts = news_data ["test_texts"]
    train_labels = news_data ["train_labels"]
    test_labels = news_data ["test_labels"]
    ag_news_labels = news_data ["ag_news_label"]

if Spam:
    spam_data = np.load ("./spam_data.npz", allow_pickle = True )
    train_texts = spam_data ["train_texts"]
    test_texts = spam_data ["test_texts"]
    train_labels = spam_data ["train_labels"]
    test_labels = spam_data ["test_labels"]
    spamlabels = spam_data ["labels"]

# Train or test data depending on user input
if Train_data_pca:
    data = train_texts
    labels = train_labels
if Test_data_pca:
    data = test_texts
    labels = test_labels
################################################

################################################
# train the fastText classifierars
if fastText:
    classifier = TextClassifier(data, labels, embed_dim=embedded_dim,
                                num_epochs=epo, ngrams=n_grams)
# get text embeddings
    Embedding = np.zeros((len(data),embedded_dim))
    for i in range(len(data)):    
        Embedding[i] = classifier.get_text_embedding(data[i])
# Get glove embeddings on the words present
if Baseline:
    global dictionary
    dictionary = {}
    
    with open("glove.6B.50d.txt", 'r', encoding='utf-8') as file:
        for line in file:
            elements = line.split();
            word = elements[0];
            vector = np.asarray(elements[1:],"float32")
            dictionary[word] = vector;
    
    def remove_unknowns(texts):
        dict_words = dictionary.keys()
        
        split_texts = [text.split(" ") for text in texts]
        
        clean_texts = []
        for text in split_texts:
            clean_text = []
            if Stopwords:
                for word in text:
                    if word in dict_words and word not in stopwords:
                        clean_text.append(word)
                if clean_text != []:
                    clean_texts.append(clean_text)
                else:
                    clean_texts.append(None)
            else:
                for word in text:
                    if word in dict_words:
                        clean_text.append(word)
                if clean_text != []:
                    clean_texts.append(clean_text)
                else:
                    clean_texts.append(None)
        return np.array(clean_texts)

    def texts_to_glove(texts):
        Z = np.zeros((len(texts), 50))
        for i, text in enumerate(texts):
            z = np.zeros((len(text),50))
            for j, word in enumerate(text):
                z[j,:] = dictionary[word]
            
            Z[i,:] = np.mean(z, axis=0)
        return Z
    rem_data = remove_unknowns(data)
    clean_labels = labels[rem_data!= None]
    clean_data = rem_data[rem_data!= None]
    
    Embedding = texts_to_glove(clean_data)
        
        
################################################

################################################
# Get embedding Vectors by transposing the text embedding matrix
vectors = Embedding.T

# Normalize vectors by subtracting mean
mean_vector = np.mean(vectors,axis=1)
data = vectors - mean_vector[:,None]

# Compute the covariance matrix
S = np.cov(data)

# Obtain eigenvectors and eigenvalues
eigenValues, eigenVectors = eigh(S)

# Sort according to size of eigenvalues
eigenValues = eigenValues[::-1]
eigenVectors = eigenVectors[:, ::-1]
Y = (eigenVectors.T@data).T[:,:2]
# Eigenvectors for baseline
if fastText:
    Y_spam = Y[labels==1]
    Y_notspam = Y[labels==0]
# Eigenvectors for baseline
if Baseline:
    Y_spam = Y[clean_labels==1]
    Y_notspam = Y[clean_labels==0]
################################################

################################################
plt.style.use("ggplot")

# Plot according to classes (2 for spam, 4 for news)
if News:
    c = ["b","r","c","m"]
    for i in range(4):
        plt.scatter(Y[labels==i][:,0],Y[labels==i][:,1],
                    label=ag_news_labels[i],c=c[i])
    plt.legend()
    plt.title(Title)
    plt.show()

if Spam:
    plt.scatter(Y_notspam[:,0],Y_notspam[:,1],label="Ham",c="c")
    plt.scatter(Y_spam[:,0],Y_spam[:,1],label="Spam",c="b")
    plt.legend()
    plt.title(Title)
    plt.show()




