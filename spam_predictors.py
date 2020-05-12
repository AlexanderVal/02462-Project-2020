######## SPAM PREDICTORS ##########
# By Felix Burmester, Alexander Valentini and Anton Jørgensen
###################################
# Import libraries
from text_classifier_main_final import TextClassifier
import numpy as np
###################################

###################################
# Load data
spam_data = np.load ("./spam_data.npz", allow_pickle = True )
train_texts = spam_data ["train_texts"]
test_texts = spam_data ["test_texts"]
train_labels = spam_data ["train_labels"]
test_labels = spam_data ["test_labels"]
spamlabels = spam_data ["labels"]
###################################

###################################
# Get predictions
classifier = TextClassifier(test_texts, test_labels, embed_dim=15,
                                 num_epochs=5, ngrams=3)

tokens = np.array(classifier.vocab.itos)
prediction_prob = np.zeros((len(tokens),2))
for i in range(len(tokens)):
    prediction_prob[i,:] = classifier.predict(tokens[i], return_prob=True)
    
###################################

###################################
# Create top 10 predicting tokens for spam and ham
sort_idx = np.argsort(prediction_prob[:,0])

tokens = tokens[sort_idx]
print("Ham predictors: ",tokens[-10:])
print("\nSpam predictors: ", tokens[:10])
###################################

###################################
# Get correctly classified spam text
prediction = np.zeros(len(test_labels))
for i in range(len(test_labels)):
    prediction[i] = classifier.predict(test_texts[i], return_prob=False)

Correct_spam_text = np.array([])
for i in range(len(test_labels)):
    if prediction[i] == test_labels[i] and test_labels[i] == 1:
        Correct_spam_text = np.append(test_texts[i],Correct_spam_text)
# Choose one spam text to edit
Worst_spam_text = Correct_spam_text[0]
print("\n",Worst_spam_text)
###################################

###################################
# Edit the spam text
Worst_spam_text = "this is the second time we have tried 2 contact you. hope i can get 2 you now. you have a £750 pound reward waiting, claim it by dialing 087187272008. only 10 p per minute bt-national-rate."

print("\nNew text; ",Worst_spam_text)

print("\nPredicted class: ",classifier.predict(Worst_spam_text))
if classifier.predict(Worst_spam_text) == 0:
    print("That equals to ham, mission accomplished")
else:
    print("Still spam, try again")



