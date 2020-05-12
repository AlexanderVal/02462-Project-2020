######## BAYESIAN CLASSIFIER ##########
# By Felix Burmester, Alexander Valentini and Anton JÃ¸rgensen
################################################


import numpy as np
from scipy.stats import multivariate_normal
from seaborn import heatmap
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

#print(stopwords.words('english'))

np.seterr(divide = "ignore")

# LOAD DICTIONARY
filename = "glove.6B.50d.txt"
D = 50

global dictionary
dictionary = {}

with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        elements = line.split();
        word = elements[0];
        vector = np.asarray(elements[1:],"float32")
        dictionary[word] = vector;
#print(len(dictionary["man"]))
# LOAD DATA
news_data = np.load("./news_data.npz", allow_pickle = True )
spam_data = np.load("./spam_data.npz", allow_pickle = True )

n_train_texts = news_data["train_texts"]
n_test_texts = news_data["test_texts"]
n_train_labels = news_data["train_labels"]
n_test_labels = news_data["test_labels"]
ag_news_labels = news_data["ag_news_label"]

s_train_texts = spam_data["train_texts"]
s_test_texts = spam_data["test_texts"]
s_train_labels = spam_data["train_labels"]
s_test_labels = spam_data["test_labels"]
spam_labels = spam_data["labels"]

print("Data Loaded")

#If true, all stopwords, punctuations and numbers will be removed. 
corpus_clean = True


#The stopwords are found in the NLTK "stopwords" package, which needs to be downloaded in order to run the cleaning function
#Run the following lines and download the package from the launcher: 
#>>> import nltk
#>>> nltk.download()

stop_words = set(stopwords.words('english'))

punctuations = [".",".",":",";","-""..."]

# Seed random values
np.random.seed(0)

#Removes unknown tokens, and if corpus_clean is chosen, also removes numbers, punctuations and stopwords. 
def remove_unknowns(texts):
    dict_words = dictionary.keys()

    #The texts are split into individual words: 
    split_texts = [text.split(" ") for text in texts]
    
    clean_texts = []
    for text in split_texts:
        clean_text = []
        for word in text:
            if word in dict_words:
                if corpus_clean:
                    if word not in stop_words:
                        if word not in punctuations:
                            if not word.isnumeric():                        
                                clean_text.append(word)
                else: 
                    clean_text.append(word)

#The cleaned texts are gatnered: 
        if clean_text != []:
            clean_texts.append(clean_text)
        else:
            clean_texts.append(None)
    return np.array(clean_texts)


def texts_to_glove(texts):
#The z embedding for each text is found as the average of all word embeddings. 
#These are stored in the matrix Z: 
    Z = np.zeros((len(texts), D))
    for i, text in enumerate(texts):
        z = np.zeros((len(text),D))
        for j, word in enumerate(text):
            z[j,:] = dictionary[word]
        z = np.mean(z, axis=0)
        Z[i,:] = z
    return Z

def fit_bayes_parameters(Ztrain, labels):
    # priors
    num, count = np.unique(labels, return_counts=True)
    priors = np.array(count) / np.size(labels)
    
    n_classes = np.size(num)
    train_classes = [0] * n_classes 
    means = [0] * n_classes
    covs = [0] * n_classes
    
    # split training set into classes
    for i in range(n_classes):
        train_classes[i] = Ztrain[labels == i, :].T
        
    # fit multivariate normal dist for each class
    for i in range(n_classes):
        means[i] = np.mean(train_classes[i], axis = 1)
        covs[i] = np.cov(train_classes[i])
        
    return means, covs, priors


def classify(z, model_params):
    means, covs, priors = model_params
    posteriors = np.empty(np.size(priors))
    for i in range(np.size(priors)):
        posteriors[i] = np.exp(
            np.log(multivariate_normal.pdf(z, means[i], covs[i])) + 
            np.log(priors[i]))
    prediction = np.argmax(posteriors)
    return prediction


# remove unkown tokens
clean_train_news = remove_unknowns(n_train_texts)
clean_test_news = remove_unknowns(n_test_texts)
clean_train_spam = remove_unknowns(s_train_texts)
clean_test_spam = remove_unknowns(s_test_texts)
print("Cleaned_data")


# remove empty texts
s_train_labels = s_train_labels[clean_train_spam != None]
clean_train_spam = clean_train_spam[clean_train_spam != None]

s_test_labels = s_test_labels[clean_test_spam != None]
clean_test_spam = clean_test_spam[clean_test_spam != None]
print("empty texts removed")


Ztrain_news = texts_to_glove(clean_train_news)
Ztest_news = texts_to_glove(clean_test_news)
Ztrain_spam = texts_to_glove(clean_train_spam)
Ztest_spam = texts_to_glove(clean_test_spam)


news_params = fit_bayes_parameters(Ztrain_news, n_train_labels)
spam_params = fit_bayes_parameters(Ztrain_spam, s_train_labels)



accuracies = np.zeros(np.size(n_train_labels))
for i, z in enumerate(Ztrain_news):
    predicted_label = classify(z, news_params)
    if predicted_label == n_train_labels[i]:
        accuracies[i] = 1
print("News, train accuracy:", np.mean(accuracies))

# TEST ACCURACY + CMATRIX
news_accuracies = np.zeros(np.size(n_test_labels))
news_cmatrix = np.zeros((4,4))
for i, z in enumerate(Ztest_news):
    predicted_label = classify(z, news_params)
    news_cmatrix[predicted_label, n_test_labels[i]] += 1
    if predicted_label == n_test_labels[i]:
        news_accuracies[i] = 1
print("News, test accuracy:", np.mean(news_accuracies))

accuracies = np.zeros(np.size(s_train_labels))
for i, z in enumerate(Ztrain_spam):
    predicted_label = classify(z, spam_params)
    if predicted_label == s_train_labels[i]:
        accuracies[i] = 1
print("Spam, train accuracy:", np.mean(accuracies))

# TEST ACCURACY + CMATRIX
spam_accuracies = np.zeros(np.size(s_test_labels))
spam_cmatrix = np.zeros((2,2))
for i, z in enumerate(Ztest_spam):
    predicted_label = classify(z, spam_params)
    spam_cmatrix[predicted_label, s_test_labels[i]] += 1
    if predicted_label == s_test_labels[i]:
        spam_accuracies[i] = 1
print("Spam, test accuracy:", np.mean(spam_accuracies))

print(news_cmatrix)
print(spam_cmatrix)

plt.figure(figsize = (6,6))
heatmap(news_cmatrix.astype("int"), square=True, annot=True, cmap="Blues", 
        xticklabels=ag_news_labels, yticklabels=ag_news_labels,
        cbar_kws = {"fraction":0.046, "pad":0.04}, fmt="d", vmin=0, vmax=2000)
plt.xlabel("True Label", fontsize = 12)
plt.ylabel("Predicted Label", fontsize = 12)
plt.title("GloVe classifier, Cleaned News Data")
plt.show()
plt.figure(figsize = (6,6))
heatmap(spam_cmatrix.astype("int"), square=True, annot=True, cmap="Blues", 
        xticklabels=spam_labels, yticklabels=spam_labels,
        cbar_kws={"fraction":0.0455, "pad":0.04}, fmt="d", vmin=0, vmax=1000)
plt.xlabel("True Label", fontsize = 12)
plt.ylabel("Predicted Label", fontsize = 12)
plt.title("GloVe classifier, Cleaned Spam Data")
plt.show()
