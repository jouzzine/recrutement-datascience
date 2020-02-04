# Imported Libraries
import matplotlib
matplotlib.use('TkAgg')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# Classifier Libraries
from sklearn.linear_model import LogisticRegression

# Other Libraries
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

###########################################################################################
############################## DATA UNDERSTANDING #########################################
###########################################################################################

# Data Recovery
df = pd.read_csv('creditcard.csv')
print('Data looks like :\n', df.head())

# Let's check if a column present missing values
print('Data have a maximum of '+str(df.isnull().sum().max())+'empty values for each column. 0 means no empty value in data.')

# Columns names
columns = df.columns
print(columns)

# Let's check the proportion of frauds and  not fraud
# It's important to know how unbalanced are the fraud and not fraud data. Most of transaction are not frauds.
# If we use all this dataset as a base of the prediction, we will get a lot of errors since it will considere most of
# transaction are not frauds, if it's none. There will be an overfitting algorithm. Moreover, we want to find pattern
# that turn the transaction fraudulent.
print('No Frauds '+ str(round(df['Class'].value_counts()[0] * 100/len(df),5))+ '% of the dataset')
print('Frauds '+ str(round(df['Class'].value_counts()[1] * 100/len(df),5))+ '% of the dataset')

# Let's see the distribution of data
fig, ax = plt.subplots(6, 5, figsize=(20,20))
for i in range(30):
    col = columns[i]
    val = df[col].values
    sns.distplot(val, ax=ax[(i//5),(i%5)])
    ax[(i//5),(i%5)].set_title('Distribution of Transaction '+col, fontsize=9)
    ax[(i//5),(i%5)].set_xlim([min(val), max(val)])
plt.show()

###########################################################################################
############################## DATA PREPROCESSING #########################################
###########################################################################################
# We should scale the columns that are least scaled (Amount and Time)
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)
columns = df.columns
# Amount and Time are Scaled!

# Now, we have to subsample train data.

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:len(fraud_df)]
new_df = pd.concat([fraud_df, non_fraud_df])
new_df = new_df.sample(frac=1, random_state=42)
print('No Frauds '+ str(round(new_df['Class'].value_counts()[0] * 100/len(new_df), 5))+ '% of the subdataset')
print('Frauds '+ str(round(new_df['Class'].value_counts()[1] * 100/len(new_df), 5))+ '% of the subdataset')

# Correation matrix
f, ax = plt.subplots(1, 1, figsize=(30,30))
sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
ax.set_title('Correlation Matrix', fontsize=14)
plt.show()

# Box plot
colors = ["#0101DF", "#DF0101"]
fig, ax = plt.subplots(6, 5, figsize=(30,30))
for i in range(30):
    col = columns[i]
    val = new_df[col].values
    sns.boxplot(x="Class", y=col, data=new_df, palette = colors, ax=ax[(i//5),(i%5)])
    ax[(i//5),(i%5)].set_title(col+' VS fraud class', fontsize=9)
plt.show()

###########################################################################################
##################################### MODEL ###############################################
###########################################################################################
X = df.iloc[:,1:23]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(sampling_strategy='minority', random_state=27)
X_train, y_train = sm.fit_sample(X_train, y_train)

model = LogisticRegression()
model.fit(X_train, y_train)
log_reg_params = {"penalty": ['l1', 'l2'], 'C': np.logspace(-4, 4, 20)}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_
print('Length of X (train): {} | Length of y (train): {}'.format(len(X_train), len(y_train)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(X_test), len(y_test)))

###########################################################################################
#################################### VALIDATION ###########################################
###########################################################################################
training_score = cross_val_score(model, X_train, y_train, cv=5)
print("LOG REG has a training score of "+ str(round(training_score.mean(), 2) * 100)+ "% accuracy score")

training_score_best_estimator = cross_val_score(log_reg, X_train, y_train, cv=5)
print("LOG REG has a training score of "+ str(round(training_score_best_estimator.mean(), 2) * 100)+ "% accuracy score")

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
plt.figure(figsize=(12, 8))
plt.title('Logistic Regression ROC Curve', fontsize=16)
plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.show()

# Confusion matrix, f1-score, precision, recall
f, ax = plt.subplots(1, 1, figsize=(30,30))
y_pred_log_reg = log_reg.predict(X_test)
log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
sns.heatmap(log_reg_cf, ax=ax, annot=True)
ax.set_title("Logistic Regression Confusion Matrix", fontsize=14)
plt.show()

print(classification_report(y_test, y_pred_log_reg))