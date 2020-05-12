######## FASTTEXT CLASSIFIER ##########
# By Felix Burmester, Alexander Valentini and Anton JÃ¸rgensen
###################################

import numpy as np
from scipy.stats import multivariate_normal
from seaborn import heatmap
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from text_classifier_main_final import TextClassifier


#Decides what part of the code to run. 
#The first option run a classification of the training data with different embedding dimensions and do k-fold validation.  
#Afterwards a plot with the dimensions and corresponding accuracies are plotted. 
#The second option will run one classification for each dimension and test the classifier on the test data: 
#Afterwards confusion matrices with the predicted and true labels are plotted.
test_options =["test_embed_dim","heatmap"]

test = test_options[1]

#Seeding the data: 
np.random.seed(0)

Corpus_clean = True


#The stopwords are found in the NLTK "stopwords" package, which needs to be downloaded in order to run the cleaning function
#Run the following lines and download the package from the launcher: 
#>>> import nltk
#>>> nltk.download()

stop_words = set(stopwords.words('english'))

punctuations = [".",",",":",";","-","..."]
#Remember to set the number of ngrams used inside the file text_classifier_main_final 


#Decides the amount of epochs used in the fasttext classifier: 
epochs = 5

# Loads the news and spam data:
spam_data = np.load("./spam_data.npz", allow_pickle = True)
s_train_texts = spam_data["train_texts"]
s_test_texts = spam_data ["test_texts"]
s_train_labels = spam_data ["train_labels"]
s_test_labels = spam_data ["test_labels"]
spam_labels = spam_data ["labels"]

news_data = np.load("./news_data.npz", allow_pickle = True )
n_train_texts = news_data["train_texts"]
n_test_texts = news_data["test_texts"]
n_train_labels = news_data["train_labels"]
n_test_labels = news_data["test_labels"]
ag_news_labels = news_data["ag_news_label"]

print("Data Loaded")


#Removes stopwords, punctuations and numbers from the data
def corpus_cleaner(texts):

    #Splits the texts into individual words
    split_texts = [text.split(" ") for text in texts]
    
    clean_texts = []
    for text in split_texts:
        clean_text = []
        
        for word in text:
            if word not in stop_words and word not in punctuations and not word.isnumeric():
        
        #Combining the individual words into one text                
                clean_text.append(word)                
        clean_text = " ".join(clean_text)            

        if clean_text != "":
            clean_texts.append(clean_text)
        #In case the text is empty after cleaning it gets added as "None"    
        else:
            clean_texts.append(None)
    return np.array(clean_texts)



#This section of the code corresponds to the first choice and tests the training accuracy for a number of dimensions through k-fold validation: 
if test == test_options[0]:
    
    data_options = ["SPAM","NEWS"]

#User decision to classify news or spam data. Choice 0 is for spam data, and choice 1 is news data.     
    data_choice = data_options[1]
    
    #Defines the used data according to the user choice: 
    if data_choice == data_options[0]: 
        data = s_train_texts
        labels = s_train_labels

    if data_choice == data_options[1]: 
        data = n_train_texts
        labels = n_train_labels

    
    # Creates a random permutation to shuffle the data: 
    N = len(data)
    randperm = np.random.permutation(N)

    # Create cross validation (5-fold)
    cross_val = 5

    #Splits the traning data into "training" and "test" data according to the chosen amount of folds: 
    test_idx = np.array([randperm[i*N//cross_val:(i+1)*N//cross_val] for i in range(cross_val)])
    train_idx = np.array([np.delete(randperm,test_idx[i]) for i in range(cross_val)])    
    
    #Dimensions for which the classifier is tested. 
    embed_dims = [75]
    embed_accuracies = np.zeros(len(embed_dims))
    for k, dim in enumerate(embed_dims):
    
        dim_accuracy = np.zeros(cross_val)
        for fold in range(cross_val):
        
            #Defining the appropriate data for the current iteration of the cross validation: 
            train_texts = data[train_idx[fold]]
            train_labels = labels[train_idx[fold]]
        
            test_texts = data[test_idx[fold]]
            test_labels = labels[test_idx[fold]]
        
            # training the classifier
            classifier = TextClassifier(train_texts, train_labels, dim, num_epochs=5)
        
            accuracy = np.zeros(len(test_texts))
            for i in range(test_texts.shape[0]):
            
                # Making a prediction of the label based on the classifier: 
                output_label = classifier.predict(test_texts[i])
            
                # calculate accuracy
                if output_label == test_labels[i]:
                    accuracy[i] = 1
            #Calculates the mean accuracy for the current iteration of the cross validation: 
            dim_accuracy[fold] = np.mean(accuracy)
            #The mean accuracy for the dimension: 
        embed_accuracies[k] = np.mean(dim_accuracy)
        print("embed_dim: ",dim)
        print("dim_accuracy: ",np.mean(dim_accuracy))

    print("List with accuracies: ", embed_accuracies)
    
    #Plots the dimensions and corresponding accuracies
    plt.style.use('ggplot')
    plt.plot(embed_dims,embed_accuracies,"bo-")
    plt.title("Embedded dimension accuracies - "+data_choice)
    plt.xlabel("Dimension of embedded vectors")
    plt.ylabel("Accuracy in proportion")
    plt.show()




#This section corresponds to the second testing choice. 


if test == test_options[1]:

    if Corpus_clean:
        #The texts are cleaned
        n_train_texts  = corpus_cleaner(n_train_texts)
        n_test_texts = corpus_cleaner(n_test_texts)
        s_train_texts = corpus_cleaner(s_train_texts)
        s_test_texts= corpus_cleaner(s_test_texts)

        # remove empty texts
        s_train_labels = s_train_labels[s_train_texts != None]
        s_train_texts = s_train_texts[s_train_texts != None]

        s_test_labels = s_test_labels[s_test_texts != None]
        s_test_texts = s_test_texts[s_test_texts != None]
                

        print("Cleaning complete")
    
    s_embed_dims = [15]
    n_embed_dims = [40]

#The classification process of the training and test data are divided into their own main loops so a different number of dimensions can be tested for each data type.     
#Loop over chosen spam dimensions: 
    for k_spam,dim_spam in enumerate (s_embed_dims):   
#Training the classifiers:

             
        spam_classifier = TextClassifier(s_train_texts, s_train_labels, dim_spam, num_epochs =epochs)

        spam_prediction_validities = np.zeros(len(s_test_texts))

        #Constructing the confusion matrix based on predicted and true labels:     
        spam_cmatrix = np.zeros((2,2))
        for i, text in enumerate(s_test_texts):
            #Predicting the label
            predicted_label = spam_classifier.predict(text)
            spam_cmatrix[predicted_label, s_test_labels[i]]  += 1
            if predicted_label == s_test_labels[i]:
                spam_prediction_validities[i] = 1
        print("Spam, test accuracy:", np.mean(spam_prediction_validities))

        #Plotting the heatmaps for the spam data: 
        plt.figure(figsize = (6,6))
        heatmap(spam_cmatrix.astype("int"), square=True, annot=True, cmap="Blues", 
                xticklabels=spam_labels, yticklabels=spam_labels,
                cbar_kws={"fraction":0.0455, "pad":0.04}, fmt="d", vmin=0, vmax=1000)
        plt.xlabel("True Label", fontsize = 12)
        plt.ylabel("Predicted Label", fontsize = 12)
        plt.title("fastText {}D, Spam Data".format(dim_spam))
        plt.show()
#Loop over news dimensions: 
    for k_news,dim_news in enumerate (n_embed_dims):
        news_classifier = TextClassifier(n_train_texts, n_train_labels, dim_news, num_epochs = epochs)
        
        
        news_prediction_validities = np.zeros(len(n_test_texts))
        wrong_spam = {"spam": [], "not_spam": []}
        #Constructing the confusion matrix: 
        news_cmatrix = np.zeros((4,4))
        for i, text in enumerate(n_test_texts):
            #Predicting the label
            predicted_label = news_classifier.predict(text)
            news_cmatrix[predicted_label, n_test_labels[i]] += 1
            if predicted_label == n_test_labels[i]:
                news_prediction_validities[i] = 1
        print("News, test accuracy:", np.mean(news_prediction_validities))
        
        # Plotting heatmaps (confusion matrices)for news data: 
        
        
        plt.figure(figsize = (6,6))
        heatmap(news_cmatrix.astype("int"), square=True, annot=True, cmap="Blues", 
                xticklabels=ag_news_labels, yticklabels=ag_news_labels,
                cbar_kws={"fraction":0.0455, "pad":0.04}, fmt="d", vmin=0, vmax=2000)
        plt.xlabel("True Label", fontsize = 12)
        plt.ylabel("Predicted Label", fontsize = 12)
        plt.title("fastText {}D, News Data".format(dim_news))
        plt.show()



