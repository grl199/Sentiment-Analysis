

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score,f1_score,zero_one_loss,confusion_matrix,precision_score,plot_confusion_matrix, log_loss

#Load and put together Henning's files
embedded_data = pd.read_csv('embedded_train_fine_tuned.csv',sep='|')
labels = pd.read_csv('data_train_train.csv',sep='|')['label']
data = pd.concat([embedded_data,labels],axis=1)
X_train=data.iloc[:,1:-1].values
y_train=data.iloc[:,-1].values


#Load and put together Henning's files
embedded_data = pd.read_csv('embedded_valid_fine_tuned.csv',sep='|')
labels = pd.read_csv('data_train_valid.csv',sep='|')['label']
data = pd.concat([embedded_data,labels],axis=1)
X_valid=data.iloc[:,1:-1].values
y_valid=data.iloc[:,-1].values

'''
#Map labels onto {1,2,3,4,5}
y[y<=0.2]=1
y[y<=0.4]=2
y[y<=0.6]=3
y[y<=0.8]=4
y[y<1]=5
'''

'''
# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)
'''

#Standarization (important for PCA, LDA, SVM,...)
from sklearn.preprocessing import StandardScaler
esc=StandardScaler()
X_train = esc.fit_transform(X_train)
X_valid = esc.transform(X_valid)

# PCA : Assesss variables' relevance
from sklearn.decomposition import PCA


pca = PCA(n_components=2,random_state=0)
pca.fit(X_train)

#Check what percentage of the variance is explained
explained_variance=[]
for i in range(500):
    explained_variance.append(sum(pca.explained_variance_ratio_[:i]))

plt.plot(explained_variance)

def pca_reduce(k,dataset,random_state=0):
    '''
    Reduce the dimension of the dataset. Returns a pca-transformed dataset
    with k features
    '''
    pca=PCA(n_components=k,random_state=random_state)
    return pca.fit_transform(dataset)
    

#Standarization (important for PCA, LDA, SVM,...)
from sklearn.preprocessing import StandardScaler
esc=StandardScaler()


#Take a (sufficiently large) random sample
N=10000
np.random.seed(0)
indexes=np.random.choice(data.shape[0],size=N)
X_train=esc.fit_transform(data.iloc[indexes,1:-1].values)
y_train= data.iloc[indexes,-1].values
X_train=esc.fit_transform(data.iloc[:,1:-1].values)
y_train= data.iloc[:,-1].values
X_train_reduced = pca_reduce(2,X_train)
y_train_reduced = y_train



plt.plot(X_train_reduced[y_train_reduced==1,0],X_train_reduced[y_train_reduced==1,1],'b*',alpha=0.25)
plt.plot(X_train_reduced[y_train_reduced==2,0],X_train_reduced[y_train_reduced==2,1],'r*',alpha=0.25)
plt.plot(X_train_reduced[y_train_reduced==3,0],X_train_reduced[y_train_reduced==3,1],'g*',alpha=0.25)
plt.plot(X_train_reduced[y_train_reduced==4,0],X_train_reduced[y_train_reduced==4,1],'k*',alpha=0.25)
plt.plot(X_train_reduced[y_train_reduced==5,0],X_train_reduced[y_train_reduced==5,1],'y*',alpha=0.25)




def scatter_2d_label(X_2d, y, ax=None, s=2, alpha=0.5, lw=2):
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
    colors = sns.color_palette(n_colors=targets.size)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # scatter plot    
    for color, target in zip(colors, targets):
        ax.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=target, s=s, alpha=alpha, lw=lw)
    
    # add legend
    ax.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3, frameon=False); # Add a legend outside the plot at specified point
    
    return ax


fig, ax = plt.subplots()
scatter_2d_label(X_train_reduced, y_train_reduced, ax)
ax.set(title='Labelled data in 2-D PCA space', 
       xlabel='Principal component score 1',
       ylabel='Principal component score 2',alpha=0.1);
#ax.legend().loc = 'best'  # if you want to place the legend elsewhere




'''
Kernel pca
'''

from sklearn.decomposition import KernelPCA

# Your code goes here

kernels = ['poly', 'rbf', 'cosine', 'sigmoid']
fig, ax = plt.subplots(2,2,figsize=(12,12));



for ii, kernel in enumerate(kernels):
    X_kpca_2d = KernelPCA(n_components=2, kernel=kernel).fit_transform(X_train)
    cur_ax = ax[ii//2, ii%2]
    scatter_2d_label(X_kpca_2d, y_train_reduced, ax=cur_ax)
    cur_ax.set(title='{} kernel'.format(kernel))
    cur_ax.legend().set_visible(False)

ax[0, 0].set_ylabel('Principal component 2')
ax[1, 0].set_ylabel('Principal component 2')

ax[1, 0].set_xlabel('Principal component 1')
ax[1, 1].set_xlabel('Principal component 1')

plt.legend(loc='center left', bbox_to_anchor=[1.01, 1.], scatterpoints=3);

    

'''
MDS
'''

# Your code goes here

from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=0, max_iter=10)
X_mds_2d = mds.fit_transform(X_train_reduced)

scatter_2d_label(X_mds_2d, y_train_reduced)
plt.title('Metric MDS, stress: {}'.format(mds.stress_))
plt.xlabel('Component 1')
plt.ylabel('Component 2');

from sklearn.metrics import accuracy_score,f1_score,zero_one_loss,confusion_matrix,precision_score,plot_confusion_matrix


from sklearn.metrics import accuracy_score
def scores_pca(clf,components):
    '''
    Check scores for several PCA dimensionality reduction
    The function uses accuracy as metric
    INPUT:
        * clf: classifier to train
        * components: array with pca components
    OUTPUT:
        * scores: array of scores (accuracy) for each number of components
    '''

    scores=[]
    for c in components :
        print('Number of components: '+str(c))
        
        #Compute the pca-transformed training and validation sets
        X_train_pca = pca_reduce(c,X_train)
        X_valid_pca = pca_reduce(c,X_valid)
        
        clf.fit(X_train_pca,y_train)
        scores.append(accuracy_score(y_valid,clf.predict(X_valid_pca)))
        
    return scores


#Example of the previous function with a Gaussian naive classifier
components=[5*i for i in range(1,20)]
scores=scores_pca(GaussianNB(),components)
plt.scatter(components,scores)




def Training_Prediction(X_train,X_valid,y_train,y_valid,clfs,clfs_names,pca,n_components):
    '''
    Function that carries out the training and outputs the predictions
    in form of dictionaries for both the training and the validation set 
    INPUT:
        * X_train : training set
        * X_valid : validation set
        * y_train : training labels
        * y_valid : validation labels
        * clfs : list with classifiers
        * clfs_names : list with the names of the classifiers
        * pca : boolean that indicates whether dimension is reduced (by PCA)
        * n_components : number of components for PCA dimensionality
          reduction (not applicable if pca=False)
    OUTPUT:
        *train_pred : dictionary cintaining classifiers as names and predictions on the
         training set as items
        *val_pred : dictionary containing classifiers as names and predictions on the
         validation set as items

    '''
    
    #Predictions
    val_pred={}
    train_pred={}

    # Train classifiers on datasets with reduced dimensionality
    # (Recommended to reduce running time)
    if pca:
        df_train = pca_reduce(n_components,X_train)
        df_valid = pca_reduce(n_components,X_valid)
    else:
        df_train = X_train
        df_valid = X_valid
        
    i=1
    for name, clf in zip(clfs_names, clfs):
        print('Model '+str(i)+"/"+str(len(clfs))+"   ("+name+")")

        clf.fit(df_train, y_train)
        train_pred[name] = clf.predict(df_train)
        val_pred[name] = clf.predict(df_valid)

        i+=1
        
    return [train_pred,val_pred]
            
 #Old list of classifiers (some of them have a bad scalability)
'''
names = ["Dummy, most frequent", "Gaussian Naive Bayes", "Logistic Regression",
         "Nearest Neighb (10)", "Nearest Neighb (5)",
         "Linear SVM", "RBF SVM",
         "Random Forest", "MLP", "MLP stronger reg", "LDA", "QDA"]
'''

names = ["Dummy, most frequent", "Gaussian Naive Bayes", "Logistic Regression",
         "Random Forest", "MLP", "MLP stronger reg", "LDA"]

classifiers = [
    DummyClassifier(strategy='most_frequent'),
    GaussianNB(),
    LogisticRegression(max_iter=100),
    #KNeighborsClassifier(n_neighbors=10),
    #KNeighborsClassifier(n_neighbors=5), 
    #SVC(kernel="linear", probability=False, random_state=0),
    #SVC(kernel='rbf', probability=False, random_state=0),
    RandomForestClassifier(max_depth=10, n_estimators=50,random_state=0),
    MLPClassifier(random_state=0, max_iter=20),  # default regularisation
    MLPClassifier(random_state=0, max_iter=20, alpha=1),  # more regularisation
    LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis()]   
    ]


names=['Softmax regression']
names=['MLPClassifier']
classifiers=[LogisticRegression(max_iter=10,verbose=100,tol=10)]
classifiers=[ MLPClassifier(random_state=0, max_iter=20,verbose=1, alpha=1)]
names=['Gaussian Naive Bayes','Dummy, most frequent','Logistic regression']
classifiers=[GaussianNB(),DummyClassifier(strategy='most_frequent'), LogisticRegression(max_iter=1000)]
classifiers=[QuadraticDiscriminantAnalysis()]

train_pred,valid_pred=Training_Prediction(X_train,X_valid,y_train,y_valid,clfs=classifiers,
                    clfs_names=names,pca=False,n_components=10)
    
    
def compute_scores(y_train,y_pred,train_pred,valid_pred):
    '''
    Calculate the scores (accuracy, weighted precision, f1 an 0-1)
    for both the training and the validation set.
    INPUT:
        * y_train : training labels
        * y_valid : validation labels
        *train_pred : dictionary cintaining classifiers as names and predictions on the
         training set as items
        *val_pred : dictionary containing classifiers as names and predictions on the
         validation set as items
    OUTPUT:
        * scores : data frame containing all the scores
    '''
    
    acc_train={}
    prec_train={}
    f1_train={}
    zero_one_train={}
    
    acc_val={}
    prec_val={}
    f1_val={}
    zero_one_val={} 
    
    for name in names:
        acc_train[name] = accuracy_score(y_train, train_pred[name])
        prec_train[name] = precision_score(y_train, train_pred[name],average='weighted')
        f1_train[name] = f1_score(y_train, train_pred[name],average='weighted')
        zero_one_train[name] = zero_one_loss(y_train, train_pred[name])
        
        acc_val[name] = accuracy_score(y_valid, valid_pred[name])
        prec_val[name] = precision_score(y_valid, valid_pred[name],average='weighted')
        f1_val[name] = f1_score(y_valid, valid_pred[name],average='weighted')
        zero_one_val[name] = zero_one_loss(y_valid, valid_pred[name])
    
    
    train_scores = pd.DataFrame(data= {'Classifier' : names ,
                                     'Set' : ['train']*len(names) ,
                                     'Accuracy' : list(acc_train.values()),
                                     'Precision' : list(prec_train.values()),
                                     'F1' : list(f1_train.values()),
                                     'zero-one' : list(zero_one_train.values())})
    
    val_scores = pd.DataFrame(data= {'Classifier' : names ,
                                     'Set' : ['validation']*len(names) ,
                                     'Accuracy' : list(acc_val.values()),
                                     'Precision' : list(prec_val.values()),
                                     'F1' : list(f1_val.values()),
                                     'zero-one' : list(zero_one_val.values())})
    
    scores= pd.concat([train_scores, val_scores],axis=0) #Axis=0 means horizontally
    scores = scores.groupby('Classifier').apply(lambda a: a[:]).iloc[:,1:]
    return scores


scores=compute_scores(y_train,y_valid,train_pred,valid_pred)
scores.to_csv('scores.csv')


clf = LogisticRegression(verbose=1,max_iter=1000,n_jobs=-1)
clf.fit(X_train,y_train)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("stsb-roberta-base")
enc=np.array(model.encode(['This was a good film']))

clf.predict(np.array(model.encode(['Did not meet my expectations'])))

clf.predict(enc)


'''
Softmax grid search
'''


clf=LogisticRegression(random_state=0, max_iter=500,tol=0.1)

from sklearn.model_selection import GridSearchCV
parameters = [{'solver': ['newton-cg','sgd','lbfgs'],
               'C' : [0.01,0.1,1,2,5,10],
              'tol' : [1]}]

from sklearn.metrics import  make_scorer
precision = make_scorer(precision_score , average='weighted')


#This is stratified cv (the split preserves the proportion of samples in each class)
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters, 
                           scoring = precision,
                           cv = 10,
                           verbose=15)


grid_search = grid_search.fit(X_train, y_train)

grid_search.best_score_

grid_search.best_params_





    