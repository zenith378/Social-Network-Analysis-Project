#!/usr/bin/env python
# coding: utf-8

# # Binary Classification
# 
# * I start with a simply binary classification problem: pro-refugees tweets vs. not-pro-refugees tweets. In order to do so I used various **classification models**. 

# In[1]:


import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from stop_words import get_stop_words

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler 

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


from sklearn.model_selection import learning_curve

#LIME
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer




# In[2]:


# stop_words = get_stop_words('english')
# stop_words.extend(['', 're', 'rt', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's', 'b',
#                   'aren', 'can', 'couldn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'let', 'mustn', 'shan', 'shouldn', 
#                    'wasn', 'weren', 'won', 'wouldn'])


# In[3]:


# utility function
def make_confusion_matrix( cfm, title):
    group_names = ['TN','FP','FN','TP']

    group_counts = ["{0:0.0f}".format(value) for value in
                cfm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                     cfm.flatten()/np.sum(cfm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    plt.title(title)
    
    sns.heatmap(cfm, annot=labels, fmt="", cmap='Blues')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted',fontsize=12)


# In[4]:


from sklearn.metrics import roc_curve,auc

def plot_roc_curve(y_test, prediction, name_model):
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, prediction)

    plt.grid()
    auc_score = round(auc(test_fpr, test_tpr),2)
    plt.plot(test_fpr, test_tpr, label=f"{name_model} - AUC ="+ str(auc_score))
    plt.plot([0,1],[0,1],'r--')
    plt.legend()
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title(f" AUC(ROC curve) - {name_model}")
    plt.grid(color='black', linestyle='', linewidth=0.5)
    plt.show()


# ## Import dataset and data preparation

# In[5]:


df = pd.read_csv('../../data_collection/data/cleaned_tweet.csv')
df = df.drop(columns=['Unnamed: 0'])


# In[6]:


unique= df["tweet_label"].unique()
freq = df["tweet_label"].value_counts()
sns.set(font_scale = 1)

ax = sns.countplot(df["tweet_label"], 
                   order = df["tweet_label"].value_counts().index)
plt.title("Target variable counts in dataset")
plt.ylabel('Number of tweets')
plt.xlabel('Tweet Type')

# adding the text labels
rects = ax.patches
for rect, label in zip(rects, freq):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
plt.show()


# In[7]:


#df = df.dropna()


# In[8]:


max_len = np.max(df['text_len'])
max_len


# In[9]:


df.sort_values(by=["text_len"], ascending=False)


# ## Dataset split

# In[10]:


data = 'Tweet_tokenized'
target = 'tweet_label'
X_train, X_test, y_train, y_test = train_test_split(df[data], df[target], test_size=0.3, random_state=42)


# In[11]:


print("x_train ->", len(X_train), "record")
print("x_test  ->", len(X_test), "record")
print("y_train ->", len(y_train), "record")
print("y_test  ->", len(y_test), "record")


# In[12]:


y_train.value_counts()


# ## Class balancing

# In[13]:


#oversampling 
ros = RandomOverSampler(sampling_strategy = 0.8)
X_train, y_train = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
train_os = pd.DataFrame(list(zip([x[0] for x in X_train], y_train)), columns = ['Tweet_tokenized', 'tweet_label'])
X_train = train_os['Tweet_tokenized'].values
y_train = train_os['tweet_label'].values


# In[14]:


#undersampling 
rus = RandomUnderSampler(sampling_strategy='majority')
X_train, y_train = rus.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
train_os = pd.DataFrame(list(zip([x[0] for x in X_train], y_train)), columns = ['Tweet_tokenized', 'tweet_label'])
X_train = train_os['Tweet_tokenized'].values
y_train = train_os['tweet_label'].values


# In[15]:


(unique, counts) = np.unique(y_train, return_counts=True)
np.asarray((unique, counts)).T


# In[16]:


sns.set(font_scale = 1)

ax = sns.barplot(unique, counts)
#plt.title("Target variable counts in balanced dataset")
plt.ylabel('Number of tweets')
plt.xlabel('Tweet Type')

#adding the text labels
rects = ax.patches
for rect, label in zip(rects, counts):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
plt.show()


# ## Tokenization

# In[17]:


idx = 1

vect = CountVectorizer(min_df = 5, ngram_range=(1,3))   # min_df: minimum number of words in a sentence
X_train_tok = vect.fit_transform(X_train)
X_test_tok =vect.transform(X_test)


# In[18]:


len(vect.vocabulary_)


# In[19]:


vect.vocabulary_


# In[20]:


vect.get_feature_names()


# In[21]:


X_train_tok[idx,:]


# In[22]:


print(X_train_tok[idx,:])


# In[23]:


vect.inverse_transform(X_train_tok[idx,:])


# In[24]:


for feat,freq in zip(vect.inverse_transform(X_train_tok[idx,:])[0],X_train_tok[idx,:].data):
    print(feat,freq)


# ## Feature selection
# Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator
# * SelectKBest removes all but takes the K highest scoring features

# In[25]:


bin_sel = SelectKBest(chi2, k = 5000).fit(X_train_tok,y_train) 
X_train_sel_bin = bin_sel.transform(X_train_tok)
X_test_sel_bin = bin_sel.transform(X_test_tok)


# In[26]:


bin_sel.get_support()


# In[27]:


X_train_sel_bin


# In[28]:


X_train_sel_bin[idx,:]


# In[29]:


print(X_train_sel_bin[idx,:])


# In[30]:


print(vect.inverse_transform(bin_sel.inverse_transform(X_train_sel_bin[idx,:])))


# ## Weigthing
# Then we apply TF-IDF transformation to associate weights to the different words based on their frequency (rarer words will be given more importance).
# 
# 

# In[31]:


tf_transformer_bin = TfidfTransformer(use_idf=True).fit(X_train_sel_bin)
X_train_tf_bin = tf_transformer_bin.transform(X_train_sel_bin)
X_test_tf_bin = tf_transformer_bin.transform(X_test_sel_bin)


# In[32]:


print(X_train_tf_bin[idx,:])


# In[33]:


for feat,weight,freq in zip(vect.inverse_transform(bin_sel.inverse_transform(X_train_tf_bin[idx,:]))[0],X_train_tf_bin[idx,:].data,X_train_sel_bin[idx,:].data):
    print(feat,weight,freq)
    


# ## Naive Bayes
# Without hyperparameter tuning because there aren't parameter to tune. 

# In[34]:


bin_nb_clf = MultinomialNB(alpha= 0.5)
bin_nb_clf.fit(X_train_tf_bin, y_train)
bin_nb_pred = bin_nb_clf.predict(X_test_tf_bin)


# In[35]:


print('Classification report:')
print(classification_report(y_test, bin_nb_pred))


# In[36]:


bin_nb_scores = cross_val_score(bin_nb_clf, X_train_tf_bin, y_train, cv=10)
cv_results_nb = cross_validate(bin_nb_clf, X_train_tf_bin, y_train, cv=10)


# In[37]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (bin_nb_scores.mean(), bin_nb_scores.std()))


# In[38]:


cv_results_nb['test_score']


# In[39]:


plot_roc_curve(y_test, bin_nb_pred, 'Naive Bayes')


# ## Linear SVC

# In[40]:


bin_svm_clf = LinearSVC().fit(X_train_tf_bin, y_train)
bin_svm_pred = bin_svm_clf.predict(X_test_tf_bin)


# In[41]:


print('Classification report:')
print(classification_report(y_test, bin_svm_pred))


# In[42]:


bin_svm_scores = cross_val_score(bin_svm_clf, X_train_tf_bin, y_train, cv=5)
print("\n \n%0.2f accuracy with a standard deviation of %0.2f" % (bin_svm_scores.mean(), bin_svm_scores.std()))


# In[43]:


bin_svm_cm = confusion_matrix(y_test, bin_svm_pred)
make_confusion_matrix(bin_svm_cm, 'Linear SVM - Confusion Matrix')


# In[44]:


plot_roc_curve(y_test, bin_svm_pred, 'Linear SVM')


# ## Logistic Regression

# In[45]:


# param_grid = {'C': [1, 10, 100], 'penalty': ["l1", "l2"],
#               "solver": ["sag", "saga"]}

# grid_lr = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid_lr.fit(X_train_tf_bin, y_train)

# print('Best Criterion:', grid_lr.best_params_)


# In[46]:


bin_lr_clf = LogisticRegression(C = 10, penalty= 'l2',
                                solver = 'sag').fit(X_train_tf_bin, y_train) 

bin_lr_pred = bin_lr_clf.predict(X_test_tf_bin)

print('Classification report:')
print(classification_report(y_test, bin_lr_pred))

bin_lr_scores = cross_val_score(bin_lr_clf, X_train_tf_bin, y_train, cv=5)
print("\n \n%0.2f accuracy with a standard deviation of %0.2f" % (bin_lr_scores.mean(), bin_lr_scores.std()))


# In[47]:


bin_lr_cm = confusion_matrix(y_test, bin_lr_pred)
make_confusion_matrix(bin_lr_cm, 'Logistic Regression - Confusion Matrix')


# In[48]:


plot_roc_curve(y_test, bin_lr_pred, 'Logistic Regression')


# ## Decision Tree

# In[49]:


# #Hyperparameter tuning
# param_grid_dt = {'criterion':['gini','entropy'], 'max_depth': np.arange(10, 40)}

# grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, refit=True, cv=5)
# grid_dt.fit(X_train_tf_bin, y_train)

# print('Best Criterion:', grid_dt.best_params_)


# In[50]:


bin_dt_clf = DecisionTreeClassifier(criterion = 'gini', max_depth=500).fit(X_train_tf_bin, y_train)
bin_dt_pred = bin_dt_clf.predict(X_test_tf_bin)

print('Classification report:')
print(classification_report(y_test, bin_dt_pred))


# In[51]:


bin_dt_cm = confusion_matrix(y_test, bin_dt_pred)
make_confusion_matrix(bin_dt_cm, 'Decision tree - Confusion Matrix')


# In[52]:


plot_roc_curve(y_test, bin_dt_pred, 'Decision Tree')


# In[53]:


bin_dt_scores = cross_val_score(bin_dt_clf, X_train_tf_bin, y_train, cv=5)
print("\n \n%0.2f accuracy with a standard deviation of %0.2f" % (bin_dt_scores.mean(), bin_dt_scores.std()))


# ## Random Forest Classifier

# In[54]:


bin_rf_clf = RandomForestClassifier(criterion = 'gini', max_depth = 500).fit(X_train_tf_bin, y_train)
bin_rf_pred = bin_rf_clf.predict(X_test_tf_bin)

print('Classification report:')
print(classification_report(y_test, bin_rf_pred))


# In[55]:


bin_rf_cm = confusion_matrix(y_test, bin_rf_pred)
make_confusion_matrix(bin_rf_cm, 'Random Forest - Confusion Matrix')


# In[56]:


plot_roc_curve(y_test, bin_rf_pred, 'Random Forest')


# In[57]:


bin_rf_scores = cross_val_score(bin_rf_clf, X_train_tf_bin, y_train, cv=5)
print("\n \n%0.2f accuracy with a standard deviation of %0.2f" % (bin_rf_scores.mean(), bin_rf_scores.std()))


# ## Calibration probabilities

# In[58]:


from sklearn.calibration import CalibrationDisplay


# In[59]:


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


# In[60]:


# Create classifiers
lr = LogisticRegression(penalty='l2', solver = 'sag', C=0.5)
gnb = MultinomialNB()
svc = NaivelyCalibratedLinearSVC(C=0.5)
rfc = RandomForestClassifier(min_samples_leaf=20, min_samples_split=30)

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (svc, "SVC"),
    (rfc, "Random forest"),
]


# In[61]:


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train_tf_bin, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test_tf_bin,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()


# # Predict all the dataset  with NAIVE-BAYES

# In[62]:


df = pd.read_csv('../../data_collection/data/dataset_all.csv')


# In[63]:


X_class = df['Tweet']


# In[64]:


#tokenization
X_class_tok =vect.transform(X_class)


# In[65]:


#feature selection
X_class_sel_bin = bin_sel.transform(X_class_tok)


# In[66]:


#weigthing
X_class_tf_bin = tf_transformer_bin.transform(X_class_sel_bin)


# In[67]:


bin_lr_pred_all = bin_lr_clf.predict(X_class_tf_bin)


# In[68]:


bin_lr_pred_all


# In[69]:


df['tweet_label'] = bin_lr_pred_all


# In[70]:


df


# In[71]:


#controlla distribuzione
import seaborn as sns

unique= df["tweet_label"].unique()
freq = df["tweet_label"].value_counts()
sns.set(font_scale = 1)

ax = sns.countplot(df["tweet_label"], 
                   order = df["tweet_label"].value_counts().index)
plt.title("Target variable counts in dataset")
plt.ylabel('Number of tweets')
plt.xlabel('Tweet Type')

# adding the text labels
rects = ax.patches
for rect, label in zip(rects, freq):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
plt.show()


# In[72]:


df.to_csv("../../data_collection/data/dataset_classified.csv")


# In[ ]:




