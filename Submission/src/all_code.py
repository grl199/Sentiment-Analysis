import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, zero_one_loss, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix

'''
Put all necessary data (sentiment_labels.txt and (dictionary.txt) into a folder called Data
locate in the same directory
'''

#Piece together and split data. Also saves the tidy data frames.
labels = pd.read_csv('Data/sentiment_labels.txt',sep='|')
phrases = pd.read_csv('Data/dictionary.txt',sep='|',header=None,names=["phrase","ids"])[["ids","phrase"]].set_index(keys="ids").sort_index()
df = pd.concat([phrases,labels["sentiment values"]],axis=1)

from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size = 0.2, random_state = 0)
df_train.to_csv("Data/tidy_train.csv",index=False,sep="|")
df_test.to_csv("Data/tidy_test.csv",index=False,sep="|")
np.sum(labels["sentiment values"]<=0.2)
np.sum(df_train["sentiment values"]<=0.2)

#Load and clean data, do some EDA
def remove_sw(phrases):
    def aux(phrase):
        phrase_tokens = phrase.split()
        filtered = [word for word in phrase_tokens if not word in stopwords.words('english')]
        return " ".join(filtered)
    phrases=phrases.apply(lambda x: aux(x))
    return phrases
def phrase_clean(phrases):
    #Takes a pandas column of phrases and cleans it :)
    def aux(phrase):
        split=phrase.split()
        if len(split)>0:
            if split[0] in set(["'ve","'s","'ll","'re","'d","'n","n't","'m"]):
                split[0]=""
                phrase=" ".join(split)
        return phrase
    #phrases = phrases.apply(lambda x: x.replace("-", " "))
    phrases = phrases.apply(lambda x: x.lower())
    phrases = phrases.apply(lambda x: re.sub(r"[^A-Za-z0-9' ]+", '', x))
    #Skip this step and let phrases begin with "'s" ect?
    phrases = phrases.apply(lambda x: x.replace("  ", " "))
    phrases = phrases.apply(lambda x: x.replace(" 've", "'ve "))
    phrases = phrases.apply(lambda x: x.replace(" 's", "'s "))
    phrases = phrases.apply(lambda x: x.replace(" 'll", "'ll "))
    phrases = phrases.apply(lambda x: x.replace(" 're", "'re "))
    phrases = phrases.apply(lambda x: x.replace(" 'd", "'d "))
    phrases = phrases.apply(lambda x: x.replace(" 'n", "'n "))
    phrases = phrases.apply(lambda x: x.replace(" 'm", "'m "))
    phrases = phrases.apply(lambda x: x.replace(" n't", "n't "))
    phrases = phrases.apply(lambda x: x.replace("  ", " "))
    phrases = phrases.apply(lambda x: x.replace("'''", ""))
    phrases = phrases.apply(lambda x: x.replace("''", ""))
    phrases = phrases.apply(lambda x: x.replace(" ' ", " "))
    phrases = phrases.apply(lambda x: x.replace(" '", " "))
    phrases = phrases.apply(lambda x: x.replace("' ", " "))
    phrases = phrases.apply(lambda x: aux(x))
    phrases = phrases.apply(lambda x: x.strip())
    return phrases
def scatter_2d_label(X_2d, y, pal, ax=None, s=1, alpha=1, lw=1):
    """Visualise a 2D embedding with corresponding labels.

    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.

    ax : matplotlib axes.Axes
         axes to plot on

    s : float
        Marker size for scatter plot.

    alpha : float
        Transparency for scatter plot.

    lw : float
        Linewidth for scatter plot.
    """

    targets = np.unique(y)  # extract unique labels
    #colors = sns.color_palette(n_colors=targets.size)

    if ax is None:
        fig, ax = plt.subplots()

    # scatter plot
    for color, target in zip(pal, targets):
        ax.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=target, s=s, alpha=alpha, lw=lw)

    # add legend
    ax.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3,
              frameon=False,markerscale=7,labels=[1,2,3,4,5]);  # Add a legend outside the plot at specified point

    return ax
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ')]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


'''
This is the full data.

data_train=pd.read_csv('tidy_train.csv',sep='|')
data_train_raw=pd.read_csv('tidy_train.csv',sep='|')
'''

#We just take a slice to make sure the code runs properly
data_train=pd.read_csv('Data/tidy_train.csv',sep='|').iloc[:100,]
data_train_raw=pd.read_csv('Data/tidy_train.csv',sep='|').iloc[:100,]

print(data_train.head(10))
#Map ranking onto a set with five different categories
data_train.loc[data_train["sentiment values"]<=0.2,"label"]=1
data_train.loc[(data_train["sentiment values"]>0.2)&(data_train["sentiment values"]<=0.4),"label"]=2
data_train.loc[(data_train["sentiment values"]>0.4)&(data_train["sentiment values"]<=0.6),"label"]=3
data_train.loc[(data_train["sentiment values"]>0.6)&(data_train["sentiment values"]<=0.8),"label"]=4
data_train.loc[(data_train["sentiment values"]>0.8)&(data_train["sentiment values"]<=1),"label"]=5
labels=data_train["label"]
sentiments=data_train["sentiment values"]



'''
General EDA
'''

#Check for imbalance
sentiments_colours=["sienna","coral","darkgray","skyblue","steelblue"]
sentiments_palette=sns.color_palette(sentiments_colours)
sns.barplot(x="label",y="label",data=data_train,
            estimator=lambda x: len(x) / len(data_train) * 100,
            palette=sentiments_palette).set(ylabel="%",xlabel="Sentiment label",xticklabels=[1,2,3,4,5])


#Check for missing values
print(data_train.isna().sum())
print(data_train.isnull().sum())


#Pre-processing (cleaning)
data_train["phrase"]=phrase_clean(data_train["phrase"])
print(data_train.head(10))
phrases=data_train["phrase"]


#Check top non-stopwords n-grams in each strata defined by label
from collections import defaultdict
#Removing stopwords takes a few minutes.
phrases_filtered=remove_sw(phrases)


#Very negative (1)
one_words=generate_ngrams(" ".join(phrases_filtered[data_train["label"]==1]),n_gram=1)
one_words_count=defaultdict(int)
for word in one_words:
    one_words_count[word]+=1
one_words_count_df = pd.DataFrame(sorted(one_words_count.items(), key=lambda x: x[1])[::-1])

#Negative (2)
two_words=generate_ngrams(" ".join(phrases_filtered[data_train["label"]==2]),n_gram=1)
two_words_count=defaultdict(int)
for word in two_words:
    two_words_count[word]+=1
two_words_count_df = pd.DataFrame(sorted(two_words_count.items(), key=lambda x: x[1])[::-1])

#Neutral (3)
three_words=generate_ngrams(" ".join(phrases_filtered[data_train["label"]==3]),n_gram=1)
three_words_count=defaultdict(int)
for word in three_words:
    three_words_count[word]+=1
three_words_count_df = pd.DataFrame(sorted(three_words_count.items(), key=lambda x: x[1])[::-1])

#Positive (4)
four_words=generate_ngrams(" ".join(phrases_filtered[data_train["label"]==4]),n_gram=1)
four_words_count=defaultdict(int)
for word in four_words:
    four_words_count[word]+=1
four_words_count_df = pd.DataFrame(sorted(four_words_count.items(), key=lambda x: x[1])[::-1])

#Very positive (5)
five_words=generate_ngrams(" ".join(phrases_filtered[data_train["label"]==5]),n_gram=1)
five_words_count=defaultdict(int)
for word in five_words:
    five_words_count[word]+=1
five_words_count_df = pd.DataFrame(sorted(five_words_count.items(), key=lambda x: x[1])[::-1])


#Plot these results
fig, axes = plt.subplots(1, 5, figsize=(13, 5))
sns.barplot(y=one_words_count_df[0].values[:20],
            x=100*one_words_count_df[1].values[:20]/sum(one_words_count_df[1]), color=sentiments_colours[0],ax=axes[0]).set(xlabel="%")
axes[0].set_title("1. Very Negative")
sns.barplot(y=two_words_count_df[0].values[:20],
            x=100*two_words_count_df[1].values[:20]/sum(two_words_count_df[1]), color=sentiments_colours[1],ax=axes[1]).set(xlabel="%")
axes[1].set_title("2. Negative")
sns.barplot(y=three_words_count_df[0].values[:20],
            x=100*three_words_count_df[1].values[:20]/sum(three_words_count_df[1]), color=sentiments_colours[2],ax=axes[2]).set(xlabel="%")
axes[2].set_title("3. Neutral")
sns.barplot(y=four_words_count_df[0].values[:20],
            x=100*four_words_count_df[1].values[:20]/np.sum(four_words_count_df[1]), color=sentiments_colours[3],ax=axes[3]).set(xlabel="%")
axes[3].set_title("4. Positive")
sns.barplot(y=five_words_count_df[0].values[:20],
            x=100*five_words_count_df[1].values[:20]/np.sum(five_words_count_df[1]), color=sentiments_colours[4],ax=axes[4]).set(xlabel="%")
axes[4].set_title("5. Very Positive")
plt.tight_layout()



#Word count in different stratas
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
data_train["count"] = data_train['phrase'].apply(lambda x: len(str(x).split()))
sns.boxplot(x="label",y="count",data=data_train,palette=sentiments_palette).set(xlabel="Label",ylabel="Count",xticklabels=[1,2,3,4,5])
print(data_train.head(10))




#What follows is the embedding. Unfortunately this takes several hours.
#Moreover, finetuning the embedder takes almost a day.

#Embedding (Sentence-BERT)
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

#Embed training (+val) data with stsb-roberta base pre-trained model
model_old = SentenceTransformer("stsb-roberta-base") #pre-trained model
train_loss_old=losses.CosineSimilarityLoss(model_old)

#We firs embed the the dataset with the pre-trained model
embedded_train_old=model_old.encode(data_train["phrase"].to_list())
embedded_train_old=pd.DataFrame(embedded_train_old)
embedded_train_old.to_csv('Data/embedded_train.csv',index=False,sep="|")


#Split training data into train+val
data_train_train, data_train_valid = train_test_split(data_train,test_size=0.2,random_state=0)
data_train_train.to_csv("Data/data_train_train.csv",index=False,sep="|")
data_train_valid.to_csv("Data/data_train_valid.csv",index=False,sep="|")

#Split train_train into pairs, each with half the observations of the training set
n = int(data_train_train.shape[0]/2)
A=data_train_train[0:n]
B=data_train_train[n:]

#Format the pairs in list along with their similarity (in sentiment)
pairs=np.hstack((A["phrase"].to_numpy().reshape((np.shape(A)[0],1)),B["phrase"].to_numpy().reshape((np.shape(A)[0],1)),
                 1-np.abs(A["sentiment values"].to_numpy().reshape((np.shape(A)[0],1))-B["sentiment values"].to_numpy().reshape((np.shape(A)[0],1)))))
train_examples=[]
for n in range(0,np.shape(A)[0]):
    train_examples.append(InputExample(texts=[pairs[n,0],pairs[n,1]],label=pairs[n,2]))
train_dataloader = DataLoader(train_examples,shuffle=True,batch_size=16)


#Fine tune stsb-roberta-base to training data
model = SentenceTransformer("stsb-roberta-base")
train_loss=losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader,train_loss)],epochs=1,warmup_steps=100)

#Save fine-tuned embedder
model.save(path="tuned_SBERT")
#embedding with tuned and saving
embedded_train=model.encode(data_train_train["phrase"].to_list())
embedded_train_df=pd.DataFrame(embedded_train)
embedded_train_df.to_csv("Data/embedded_train_fine_tuned.csv",index=False,sep="|")
embedded_valid=model.encode(data_train_valid["phrase"].to_list())
embedded_valid_df=pd.DataFrame(embedded_valid)
embedded_valid_df.to_csv("Data/embedded_valid_fine_tuned.csv",index=False,sep="|")

#Embedd the test data with fine tuned model
data_test=pd.read_csv('Data/tidy_test.csv',sep='|')
data_test["phrase"]=phrase_clean(data_test["phrase"])

#We make take a slice of the test test to accelerate the process of checking
#embedded_test=model.encode(data_test["phrase"].to_list())
embedded_test=model.encode(data_test["phrase"].iloc[:100,].to_list())

embedded_test_df=pd.DataFrame(embedded_test)
embedded_test_df.to_csv("Data/embedded_test_fine_tuned.csv",index=False,sep="|")

#Embedding the test data with stsb-roberta base pre-trained model
#We make take a slice of the test test to accelerate the process of checking
#embedded_test_old=model_old.encode(data_test["phrase"].to_list())
embedded_test_old=model_old.encode(data_test["phrase"].iloc[:100,].to_list())

embedded_test_old_df=pd.DataFrame(embedded_test_old)
embedded_test_old_df.to_csv("Data/embedded_test_old.csv",index=False,sep="|")

#Visualizing the embeddings with PCA
#Import embedded and corresponding data (for labels)
X_train_tuned=pd.read_csv('Data/embedded_train_fine_tuned.csv',sep='|')
X_valid_tuned=pd.read_csv('Data/embedded_valid_fine_tuned.csv',sep='|')
X_train_old=pd.read_csv('Data/embedded_train.csv',sep='|')

#First column in old embedded has count here...
#X_train_old=X_train_old.iloc[:,1:]


#Recover embeddings from training set (the old ones are training+val)
X_train_train_old,X_train_val_old=train_test_split(X_train_old,test_size=0.2,random_state=0)
#To centre data
sc=StandardScaler(with_std=False)
X_train_tuned_sc=sc.fit_transform(X_train_tuned)
X_train_old_sc=sc.fit_transform(X_train_train_old)
X_valid_tuned_sc=sc.fit_transform(X_valid_tuned)
X_valid_old_sc=sc.fit_transform(X_train_val_old)
#Linear PCA
pca=PCA(n_components=2)
Z_tuned=pca.fit_transform(X_train_tuned_sc)
Z_old=pca.fit_transform(X_train_old_sc)
Z_valid_tuned=pca.fit_transform(X_valid_tuned_sc)
Z_valid_old=pca.fit_transform(X_valid_old_sc)
#Plot the PCA-scores
scatter_2d_label(Z_tuned,data_train_train["label"],pal=sentiments_palette,s=0.6,lw=0).set(xlabel="PC scores 1",ylabel="PC scores 2")
scatter_2d_label(Z_old,data_train_train["label"],pal=sentiments_palette,s=0.6,lw=0).set(xlabel="PC scores 1",ylabel="PC scores 2")

scatter_2d_label(Z_valid_tuned,data_train_valid["label"],pal=sentiments_palette,s=2,lw=0).set(xlabel="PC scores 1",ylabel="PC scores 2")
scatter_2d_label(Z_valid_old,data_train_valid["label"],pal=sentiments_palette,s=2,lw=0).set(xlabel="PC scores 1",ylabel="PC scores 2")

###
### Fitting and validating the Gaussian Naive Bayes classifier (with PCA)
###
embedded_data = pd.read_csv('Data/embedded_train_fine_tuned.csv',sep='|')
labels = pd.read_csv('Data/data_train_train.csv',sep='|')['label']
X_train=embedded_data
y_train=labels
#Load and put together Henning's files
embedded_data = pd.read_csv('Data/embedded_valid_fine_tuned.csv',sep='|')
labels = pd.read_csv('Data/data_train_valid.csv',sep='|')['label']
X_valid=embedded_data
y_valid=labels

embedded_old = pd.read_csv('Data/embedded_train.csv',sep='|')
labels_old = pd.read_csv('Data/data_train_valid.csv',sep='|')['label']
#X_old=embedded_old.iloc[:,1:] #First column is count...

#We used random state 0 for split
X_old_train,X_old_valid=train_test_split(embedded_old,test_size=0.2,random_state=0)

valid_accuracy_old=[]
train_accuracy_old=[]
valid_f1_old=[]
train_f1_old=[]
kk=list(set(np.exp(np.linspace(0,4,30)).round(0)))
for k in kk:
    pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=int(k))), ("clf", GaussianNB())])
    pipe.fit(X_old_train, y_train)
    valid_accuracy_old.append(accuracy_score(y_valid,pipe.predict(X_old_valid)))
    train_accuracy_old.append(accuracy_score(y_train,pipe.predict(X_old_train)))
    valid_f1_old.append(f1_score(y_valid,pipe.predict(X_old_valid),average="weighted"))
    train_f1_old.append(f1_score(y_train,pipe.predict(X_old_train),average="weighted"))
plt.plot(valid_accuracy_old)
plt.plot(train_accuracy_old)
print("Gaussian NB (pre-trained) has training and validation accuracies",
      np.round(train_accuracy_old[np.argmax(valid_accuracy_old)],4),"and",
      np.round(valid_accuracy_old[np.argmax(valid_accuracy_old)],4),"respectively",
      "at the optimal",int(list(kk)[np.argmax(valid_accuracy_old)]),
      "PCs. The associated training and validation F1 are",
      np.round(train_f1_old[np.argmax(valid_accuracy_old)],4),
      "and",np.round(valid_f1_old[np.argmax(valid_accuracy_old)],4))

valid_accuracy_tuned=[]
train_accuracy_tuned=[]
valid_f1_tuned=[]
train_f1_tuned=[]
for k in kk:
    pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=int(k))), ("clf", GaussianNB())])
    pipe.fit(X_train, y_train)
    valid_accuracy_tuned.append(accuracy_score(y_valid,pipe.predict(X_valid)))
    train_accuracy_tuned.append(accuracy_score(y_train,pipe.predict(X_train)))
    valid_f1_tuned.append(f1_score(y_valid,pipe.predict(X_valid),average="weighted"))
    train_f1_tuned.append(f1_score(y_train,pipe.predict(X_train),average="weighted"))

plt.plot(valid_accuracy_tuned)
plt.plot(train_accuracy_tuned)
print("Gaussian NB (tuned) has training and validation accuracies",
      np.round(train_accuracy_tuned[np.argmax(valid_accuracy_tuned)],6),"and",
      np.round(valid_accuracy_tuned[np.argmax(valid_accuracy_tuned)],4),"respectively",
      "at the optimal",int(list(kk)[np.argmax(valid_accuracy_tuned)]),
      "PCs. The associated training and validation F1 are",
      np.round(train_f1_tuned[np.argmax(valid_accuracy_tuned)],4),
      "and",np.round(valid_f1_tuned[np.argmax(valid_accuracy_tuned)],4))

###
### Fitting and validating the Softmax classifier with respect to L2 regularisation penalty.
###
embedded_train=pd.read_csv('Data/embedded_train_fine_tuned.csv',sep='|')
embedded_valid=pd.read_csv('Data/embedded_valid_fine_tuned.csv',sep='|')
df_train=pd.read_csv('Data/data_train_train.csv',sep='|')
df_valid=pd.read_csv('Data/data_train_valid.csv',sep='|')
embedded_notfit=pd.read_csv('Data/embedded_train.csv',sep='|')
embedded_notfit_train,embedded_notfit_valid=train_test_split(embedded_notfit,test_size=0.2,random_state=0)
df=pd.concat((df_train,df_valid))

X_old=embedded_notfit.to_numpy()
#X_old=embedded_notfit.to_numpy()[:,1:] #first column is count....
X_new=pd.concat((embedded_train,embedded_valid)).to_numpy()

X_old_train=embedded_notfit_train.to_numpy()
#X_old_train=X_old_train[:,1:] #First column is count...
X_old_valid=embedded_notfit_valid.to_numpy()
#X_old_valid=X_old_valid[:,1:] #First column is count...
X_train=embedded_train.to_numpy()
X_valid=embedded_valid.to_numpy()
y_train=df_train["label"].to_numpy().reshape((len(df_train["label"]),))
y_valid=df_valid["label"].to_numpy().reshape((len(df_valid["label"]),))

cc=1/np.exp(np.linspace(-4,4,20))
scores_old=[]
scores_new=[]
pipe_old=Pipeline([("sc",StandardScaler()),("classifier",LogisticRegression(max_iter=2000,random_state=0,penalty='l2',multi_class="multinomial"))])
pipe_tuned = Pipeline([("sc",StandardScaler()),("classifier",LogisticRegression(max_iter=2000,random_state=0,penalty='l2',multi_class="multinomial"))])
for c in cc:
    pipe_old.steps[1][1].C=c
    pipe_old.fit(X_old_train,y_train)
    y_pred_old_train=pipe_old.predict(X_old_train)
    y_pred_old_valid=pipe_old.predict(X_old_valid)
    scores_old.append([[accuracy_score(y_train,y_pred_old_train),accuracy_score(y_valid,y_pred_old_valid)],
                       [precision_score(y_train,y_pred_old_train,average="weighted"),precision_score(y_valid,y_pred_old_valid,average="weighted")],
                       [f1_score(y_train,y_pred_old_train,average="weighted"),f1_score(y_valid,y_pred_old_valid,average="weighted")],
                       [recall_score(y_train,y_pred_old_train,average="weighted"),recall_score(y_valid,y_pred_old_valid,average="weighted")],
                       [zero_one_loss(y_train,y_pred_old_train),zero_one_loss(y_valid,y_pred_old_valid)]])
    pipe_tuned.steps[1][1].C=c
    pipe_tuned.fit(X_train, y_train)
    y_pred_tuned_train = pipe_tuned.predict(X_train)
    y_pred_tuned_valid = pipe_tuned.predict(X_valid)
    scores_new.append([[accuracy_score(y_train, y_pred_tuned_train), accuracy_score(y_valid, y_pred_tuned_valid)],
                       [precision_score(y_train, y_pred_tuned_train, average="weighted"),precision_score(y_valid, y_pred_tuned_valid, average="weighted")],
                       [f1_score(y_train, y_pred_tuned_train, average="weighted"),f1_score(y_valid, y_pred_tuned_valid, average="weighted")],
                       [recall_score(y_train, y_pred_tuned_train, average="weighted"),recall_score(y_valid, y_pred_tuned_valid, average="weighted")],
                       [zero_one_loss(y_train, y_pred_tuned_train), zero_one_loss(y_valid, y_pred_tuned_valid)]])

#scores: [...,[[training accuracy,validation accuracy],[training precision,validation precision],
#         [f1 train,f1 validation],[zero_one_loss train,zero_one_loss validation]],...]

acc_valid_old=[]
acc_valid_new=[]
prec_valid_old=[]
for i in range(0,len(cc)):
    acc_valid_old.append(scores_old[i][0][1])
    acc_valid_new.append(scores_new[i][0][1])

plt.plot(acc_valid_old)
plt.plot(acc_valid_new)
best_old=np.argmax(acc_valid_old)
best_new=np.argmax(acc_valid_new)
print("Max validation accuracy for old is:",max(acc_valid_old),"at C being:",cc[best_old])
print("Max validation accuracy for tuned is:",max(acc_valid_new),"at C being:",cc[best_new])
results_old= {"acc_train": scores_old[best_old][0][0],"acc_val": scores_old[best_old][0][1],
              "prec_train": scores_old[best_old][1][0],"prec_val": scores_old[best_old][1][1],
              "f1_train": scores_old[best_old][2][0],"f1_val": scores_old[best_old][2][1],
              "recall_train": scores_old[best_old][3][0],"recall_val": scores_old[best_old][3][1],
              "0_1_train": scores_old[best_old][4][0],"0_1_val": scores_old[best_old][4][1]}
results_new= {"acc_train": scores_new[best_new][0][0],"acc_val": scores_new[best_new][0][1],
              "prec_train": scores_new[best_new][1][0],"prec_val": scores_new[best_new][1][1],
              "f1_train": scores_new[best_new][2][0],"f1_val": scores_new[best_new][2][1],
              "recall_train": scores_new[best_new][3][0],"recall_val": scores_new[best_new][3][1],
              "0_1_train": scores_new[best_new][4][0],"0_1_val": scores_new[best_new][4][1]}
print(results_old)
print(results_new)

#Clustering for the unsupervised part
def purity(y_true, y_pred):
    cm = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


embedded=pd.read_csv('Data/embedded_train.csv',sep='|')
#embedded=pd.read_csv('Data/embedded_train.csv',sep='|').iloc[:,1:] #First column is count...
labels = data_train['label']
y_true=labels.to_numpy().reshape((len(labels),))-1
kmeans=MiniBatchKMeans(n_clusters=5,random_state=0)
ypred_kmeans=kmeans.fit_predict(embedded)
print("K-means purity:",purity(y_true,ypred_kmeans))
print("Dummy purity:", purity(y_true,np.ones(shape=(len(labels,)))))

###
### Fit the best classifier to the test data and report results
###
## Load train, validation and test data
embedded_test_tuned=pd.read_csv('Data/embedded_test_fine_tuned.csv',sep='|')
data_test=pd.read_csv('Data/tidy_test.csv',sep='|')
data_test.loc[data_test["sentiment values"]<=0.2,"label"]=1
data_test.loc[(data_test["sentiment values"]>0.2)&(data_test["sentiment values"]<=0.4),"label"]=2
data_test.loc[(data_test["sentiment values"]>0.4)&(data_test["sentiment values"]<=0.6),"label"]=3
data_test.loc[(data_test["sentiment values"]>0.6)&(data_test["sentiment values"]<=0.8),"label"]=4
data_test.loc[(data_test["sentiment values"]>0.8)&(data_test["sentiment values"]<=1),"label"]=5
labels_test=data_test["label"]
embedded_train_tuned=pd.read_csv('Data/embedded_train_fine_tuned.csv',sep='|')
embedded_valid_tuned=pd.read_csv('Data/embedded_valid_fine_tuned.csv',sep='|')
data_train_train=pd.read_csv('Data/data_train_train.csv',sep='|')
data_train_valid=pd.read_csv('Data/data_train_valid.csv',sep='|')
labels_train_train=data_train_train["label"]
labels_train_valid=data_train_valid["label"]
#Combinde train and validation data
embedded_80_tuned=pd.concat([embedded_train_tuned,embedded_valid_tuned])
labels_80=pd.concat([labels_train_train,labels_train_valid])

#Prepare data for classification
X_80=embedded_80_tuned.to_numpy()
X_test=embedded_test_tuned.to_numpy()
y_80=labels_80.to_numpy().reshape((len(labels_80),))
y_test=labels_test.to_numpy().reshape((len(labels_test),))

#Produce final test results
pipe = Pipeline([("sc",StandardScaler()),("classifier",LogisticRegression(C=0.2290799498154878,max_iter=2000,random_state=0,penalty='l2',multi_class="multinomial"))])
pipe.fit(X_80,y_80)
y_pred=pipe.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
f1_test=f1_score(y_test,y_pred,average="weighted")
results_test={"Test accuracy":acc_test,"Test F1": f1_test}
print(results_test)
plot_confusion_matrix(pipe, X_test, y_test,display_labels=[1,2,3,4,5],normalize="true",cmap="Greys")
