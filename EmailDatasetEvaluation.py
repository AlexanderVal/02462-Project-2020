#email data loading taken from https://www.kaggle.com/veleon/spam-classification
import numpy as np
import os
import email
import email.policy

from text_classifier_main_final import TextClassifier
from bs4 import BeautifulSoup

spamdir = os.path.join("ham-and-spam-dataset", "spam")
hamdir = os.path.join("ham-and-spam-dataset", "ham")

ham_filenames = [name for name in sorted(os.listdir(hamdir)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(spamdir)) if len(name) > 20]


def load_email(is_spam, filename):
    directory = spamdir if is_spam else hamdir
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    
    
from collections import Counter

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


def html_to_plain(email):
    try:
        soup = BeautifulSoup(email.get_content(), 'html.parser')
        return soup.text.replace('\n\n','')
    except:
        return "empty"

def email_to_plain(email):
    struct = get_email_structure(email)
    for part in email.walk():
        partContentType = part.get_content_type()
        if partContentType not in ['text/plain','text/html']:
            continue
        try:
            partContent = part.get_content()
        except: # in case of encoding issues
            partContent = str(part.get_payload())
        if partContentType == 'text/plain':
            return partContent
        else:
            return html_to_plain(part)


plain_ham = []
for em in ham_emails:
    plain = email_to_plain(em)
    if plain is not None:
        plain_ham.append(plain)

plain_spam = []
for em in spam_emails:
    plain = email_to_plain(em)
    if plain is not None:
        plain_spam.append(plain)



X = plain_ham + plain_spam
y = np.array([0] * len(plain_ham) + [1] * len(plain_spam))

#Delete erroneous emails
X = np.delete(X, [2697, 2891])
y = np.delete(y, [2697, 2891])

# =============================================================================
# emaildata = []
# for em in X:
#     plain = email_to_plain(em)
#     if plain is not None:
#         emaildata.append(plain)
# emaildata = np.array(emaildata)
# 
# =============================================================================
spam_data = np.load('spam_data.npz', allow_pickle=True)
train_texts = spam_data['train_texts']
test_texts = spam_data['test_texts']
train_labels = spam_data['train_labels']
spamtest_labels = spam_data['test_labels']
spamlabels = spam_data['labels']


embed_dim = 15

# Train model
classifier = TextClassifier(train_texts , train_labels , embed_dim , ngrams = 3, num_epochs=5)


sms_trained_preds = np.array([classifier.predict(x) for x in X])
sms_trained_accuracy = np.mean(sms_trained_preds == y)
print("SMS text model accuracy for emails:", sms_trained_accuracy)
  

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


classifier = TextClassifier(X_train , y_train.astype('int64') , embed_dim , ngrams = 3, num_epochs=5)


email_trained_preds = np.array([classifier.predict(x) for x in X_test])
email_trained_accuracy = np.mean(email_trained_preds == y_test)
print("Email text model accuracy:", email_trained_accuracy)

    











