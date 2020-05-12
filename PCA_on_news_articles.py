######## PRINCIPAL COMPONENT ANALYSIS ##########
# By Felix Burmester, Alexander Valentini and Anton JÃ¸rgensen
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
np.random.seed(0)
# Input a title for plot (fx. fastText News Test 15 dim)
Title = "fastText News Examples"

# Choose embedded vectors' dimension for fastText
embedded_dim = 40
# Choose number of epochs
epo = 5
# Choose number of ngrams
n_grams = 3
    

################################################
news_data = np.load ("./news_data.npz", allow_pickle = True )
train_texts = news_data ["train_texts"]
test_texts = news_data ["test_texts"]
train_labels = news_data ["train_labels"]
test_labels = news_data ["test_labels"]
ag_news_labels = news_data ["ag_news_label"]


################################################
# train the fastText classifierars

data = train_texts
labels = train_labels

classifier = TextClassifier(data, labels, embed_dim=embedded_dim,
                            num_epochs=epo, ngrams=n_grams)
# get text embeddings
Embedding = np.zeros((len(data),embedded_dim))
for i in range(len(data)):    
    Embedding[i] = classifier.get_text_embedding(data[i])
 
        
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


business = """Tesla CEO Elon Musk threatened Saturday to pull the company’s factory
 and headquarters out of California and sued local officials who have stopped 
 the company from reopening its electric vehicle factory.

In a lawsuit filed in federal court, Tesla accused the Alameda County Health 
Department of overstepping federal and state coronavirus restrictions when it 
stopped Tesla from restarting production at its factory in Fremont. The lawsuit 
contends Tesla factory workers are allowed to work during California's 
stay-at-home order because the facility is considered “critical infrastructure."

“Frankly, this is the final straw,” Musk tweeted. “Tesla will now move its HQ 
and future programs to Texas/Nevada immediately.”

He wrote that whether the company keeps any manufacturing in Fremont depends 
on how Tesla is treated in the future.

An order in the six-county San Francisco Bay Area forced Tesla to close the 
plant starting March 23 to help prevent the virus’ spread, and it was extended 
until the end of May. Public health experts say the orders have reduced the 
number of new coronavirus cases nationwide. California Gov. Gavin Newsom allowed 
the Bay Area counties to continue restrictions while easing them in other areas 
of the state.

But the statement also said residents and businesses have made sacrifices to 
protect the health of people in the region. “It is our collective responsibility 
to move through the phases of reopening and loosening the restrictions of the 
shelter-in-place order in the safest way possible, guided by data and science,” 
the department said.

Fremont Mayor Lily Mei wrote in a statement that she is growing concerned about 
the regional economy without provisions for major manufacturing to resume 
operations. “We know many essential businesses have proven they can successfully
 operate using strict safety and social distancing practices,” the statement said.

Emails seeking comment from Newsom have not been returned.

Despite Musk's threat, it would be costly and difficult to quickly shift production 
from Fremont to Texas or Nevada. The Fremont facility, which was formerly run jointly
 by General Motors and Toyota, currently is Tesla’s only U.S. vehicle assembly plant, 
 and the company would lose critical production if it shut down the plant to move equipment.

“Moving away from Fremont would take at least 12 to 18 months and could add risk 
to the manufacturing and logistics process in the meantime,” Wedbush Securities 
analyst Daniel Ives wrote in a note to investors.

But Musk plans another U.S. factory to increase output, possibly in Texas, and 
could move production once that plant is up and running.

The lack of production in Fremont cuts off Tesla's revenue and is a big financial 
strain. On a conference call last month, Musk said the company only has assembly 
plants in Fremont and Shanghai, and the Fremont facility produces the majority of 
its vehicles. He called the closure of Fremont a “serious risk.”


Elon Musk, CEO and product architect of Tesla, said the company is suing Alameda County 
"immediately." Christie Smith reports.
The coronavirus causes mild or moderate symptoms for most people. But it has killed more 
than 78,000 people in the U.S., with the death toll rising.

Ives wrote that there's now a high-stakes poker game between Musk and county officials — 
and Musk showed his cards. “Now all eyes move to the courts and the response from Alameda 
County and potentially California state officials.”

Musk’s tweets come as competing automakers are starting to reopen factories in the U.S. 
Toyota will restart production on Monday, while General Motors, Ford and Fiat Chrysler
 all plan to restart their plants gradually on May 18. Tesla is the only major automaker 
 with a factory in California.

Musk's threats came after a series of bizarre tweets earlier this month, including 
one that said Tesla’s stock price was too high. Musk also posted parts of the U.S. 
national anthem and wrote that he would sell his houses and other possessions."""

print("Tesla sues American state, Prediction:", 
      ag_news_labels[classifier.predict(business)], classifier.predict(business, return_prob=True))
business_embed = classifier.get_text_embedding(business)
business_y = (eigenVectors.T@business_embed.T).T[:,:2]

sports = """Mary Pratt, a member of the original 1943 Rockford Peaches of the 
All-American Girls Professional Baseball League (AAGPBL), has died. She was 101 
years old. Her death was confirmed on Wednesday by her nephew, Walter Pratt.
Pratt was a lefty pitcher for the Peaches and later with the Kenosha Comets. 
Her best season was with the 1944 Comets, finishing with 21 wins, a 2.50 ERA 
and 26 strikeouts. Pratt was a graduate of Boston University with a degree in 
physical education and she had a 40-plus year teaching career. She also spent 
time officiating basketball, softball, field hockey and lacrosse games.

"It was in 1943 that I had the opportunity to become a member of the AAGPBL," 
Pratt wrote on the AAGPBL website. "In June of that year, I was contacted by 
personnel in Chicago and flew out to Chicago after the close of school. I was
 met by Mr. Ken Sells, appointed by Mr. Philip Wrigley as President of the 
 AAGPBL. I was escorted to Rockford and joined that team. That evening,
 Rockford was in the process of playing a league game at the 15th Ave. 
 stadium. That was my introduction into the All-American and the start of 
 five wonderful summers as a member of the league, 1943-47. I was fortunate 
 to have participated during those eras."

The AAGPBL was formed in 1943 when Major League Baseball players were called 
for military service during World War II. What started as a way to keep ballparks 
busy and produce wartime entertainment, quickly progressed into a professional 
league for women baseball players.

The Rockford Peaches were one of the four original teams in the AAGPBL's 
11-year existence, playing from 1943 through 1954. Rockford was one of the 
teams featured in the 1992 film "A League of Their Own." """

print("Dead female baseball player, Prediction:", 
      ag_news_labels[classifier.predict(sports)], classifier.predict(sports, return_prob=True))
sports_embed = classifier.get_text_embedding(sports)
sports_y = (eigenVectors.T@sports_embed.T).T[:,:2]


scitec = """TikTok has become immensely popular worldwide including India. 
The short-video sharing platform, however, has not been without its share 
of controversies, especially around user privacy and security. The social 
networking company is once again under scanner. The Dutch privacy watchdog 
has said it will now probe how the Chinese company handles data of its young
users.

“For many users this is an important way of staying in touch with friends 
and spending time together, particularly during the current coronavirus 
crisis,” the Dutch Data Protection Authority (DPA) is quoted as saying. 
“The rise of TikTok has led to growing concerns about privacy.”

The watchdog said it will “examine whether TikTok adequately protects the 
privacy of Dutch children.” The DPA further said it will investigate whether 
app explicitly tells that the app needs parental consent for TikTok to collect, 
store, and use kids’ personal data.

As said earlier, TikTok has faced scrutiny around the world. For instance, 
the US Navy banned the application in December last year from its government 
issued smartphones. The app was also briefly banned in India over objectionable 
content on its platform. Last year in February, it was hit with $5.7 million 
fine in US for violating child privacy laws.

Following wide security concerns, Twitter has taken a slew of measures 
including opening a transparency center. The company has also begun publishing 
transparency report on the governmental requests for its users’ account 
information."""

print("Tiktok data monitoring, Prediction:", 
      ag_news_labels[classifier.predict(scitec)], classifier.predict(scitec, return_prob=True))
scitec_embed = classifier.get_text_embedding(scitec)
scitec_y = (eigenVectors.T@scitec_embed.T).T[:,:2]


world = """Law enforcement in Mexico's Jalisco state discovered a mass grave 
with the bodies of at least 25 unidentified people, the prosecutor's office 
in that state said.

Investigators also found five bags that may hold human remains, according to 
the Jalisco state prosecutors' office in a press release posted on its Twitter
 account late Saturday.
The remains were found in El Salto, a city southeast of Guadalajara, according
 to the prosecutor's office.
The bodies and the bags will undergo forensic analysis, the office said.
The state has seen rising violence in recent years. It's the base of Jalisco
 New Generation, one of Mexico's "most powerful and fastest growing" drug 
 cartels, according to the US Drug Enforcement Agency. It split from the 
 Sinaloa Cartel 10 years ago.
The agency said its personnel are continuing to work "despite the health 
risks due to Covid-19" in fundamental areas, including in the search for 
disappeared persons."""

print("Mexico mass grave, Prediction:", 
      ag_news_labels[classifier.predict(world)], classifier.predict(world, return_prob=True))
world_embed = classifier.get_text_embedding(world)
world_y = (eigenVectors.T@world_embed.T).T[:,:2]

noclass = """The phrase "the Fall of Rome" suggests that some cataclysmic event 
ended the Roman Empire, which stretched from the British Isles to Egypt and Iraq. 
But in the end, there was no straining at the gates, no barbarian horde that 
dispatched the Roman Empire in one fell swoop.

Instead, the Roman Empire fell slowly as a result of challenges from within 
and without, changing over the course of hundreds of years until its form was 
unrecognizable. Because of the long process, different historians have placed 
an end date at many different points on a continuum. Perhaps the Fall of Rome 
is best understood as a compilation of various maladies that altered a large 
swath of human habitation over many hundreds of years.
In his masterwork, The Decline and Fall of the Roman Empire, historian Edward 
Gibbon selected 476 CE, a date most often mentioned by historians.1﻿ That date 
was when Odoacer, the Germanic king of the Torcilingi, deposed Romulus 
Augustulus, the last Roman emperor to rule the western part of the Roman 
Empire. The eastern half became the Byzantine Empire, with its capital at
 Constantinople (modern Istanbul).

But the city of Rome continued to exist. Some see the rise of Christianity
 as putting an end to the Romans; those who disagree with that find the rise 
 of Islam a more fitting bookend to the end of the empire—but that would put 
 the Fall of Rome at Constantinople in 1453!2﻿ In the end, the arrival of 
 Odoacer was but one of many barbarian incursions into the empire. Certainly, 
 the people who lived through the takeover would probably be surprised by the 
 importance we place on determining an exact event and time.

How Did Rome Fall?
Just as the Fall of Rome was not caused by a single event, the way Rome fell 
was also complex. In fact, during the period of imperial decline, the empire 
actually expanded. That influx of conquered peoples and lands changed the 
structure of the Roman government. Emperors moved the capital away from the 
city of Rome, too. The schism of east and west created not just an eastern 
capital first in Nicomedia and then Constantinople, but also a move in the 
west from Rome to Milan.

Rome started out as a small, hilly settlement by the Tiber River in the 
middle of the Italian boot, surrounded by more powerful neighbors. By the 
time Rome became an empire, the territory covered by the term "Rome" looked 
completely different. It reached its greatest extent in the second century 
CE. Some of the arguments about the Fall of Rome focus on the geographic 
diversity and the territorial expanse that Roman emperors and their 
legions had to control.1"""

print("Fall of rome, Prediction:", 
      ag_news_labels[classifier.predict(noclass)], classifier.predict(noclass, return_prob=True))
noclass_embed = classifier.get_text_embedding(noclass)
noclass_y = (eigenVectors.T@noclass_embed.T).T[:,:2]







################################################
plt.style.use("ggplot")

# Plot according to classes (2 for spam, 4 for news)

c = ["b","r","c","m"]
articles = np.array([business_y, sports_y, scitec_y, world_y])
for i in range(4):
    plt.scatter(Y[labels==i][:,0],Y[labels==i][:,1],
                label=ag_news_labels[i],c=c[i])
for i in range(4):
    plt.scatter(articles[i,0,0], articles[i,0,1], c=c[i], s=100, marker="P", 
                label=ag_news_labels[i]+" Article", edgecolors="k")

plt.scatter(noclass_y[0,0], noclass_y[0,1], c="y", s=100, marker="P", 
            label="Historical Article", edgecolors="k")


plt.legend(ncol=2)
plt.title(Title)
plt.show()



