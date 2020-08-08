import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

"""import pandas as pd
import matplotlib.pyplot as plt"""
import numpy


data = pd.read_csv('KDDTrain+.csv')

nullData = data.isnull().sum()

print(nullData)

print(data["duration"].value_counts())

print(data["protocol_type"].value_counts())

print(data["service"].value_counts())

print(data["flag"].value_counts())

print(data["src_bytes"].value_counts())

print(data["dst_bytes"].value_counts())

print(data["land"].value_counts())

print(data["wrong_fragment"].value_counts())

print(data["urgent"].value_counts())

print(data["hot"].value_counts())

print(data["num_failed_logins"].value_counts())

print(data["logged_in"].value_counts())

print(data["num_compromised"].value_counts())

print(data["root_shell"].value_counts())
print(data["su_attempted"].value_counts())
print(data["num_root"].value_counts())
print(data["num_file_creations"].value_counts())
print(data["num_shells"].value_counts())
print(data["num_access_files"].value_counts())
print(data["num_outbound_cmds"].value_counts())
print(data["is_host_login"].value_counts())
print(data["is_guest_login"].value_counts())
print(data["count"].value_counts())
print(data["srv_count"].value_counts())
print(data["serror_rate"].value_counts())
print(data["srv_serror_rate"].value_counts())
print(data["rerror_rate"].value_counts())
print(data["srv_rerror_rate"].value_counts())
print(data["same_srv_rate"].value_counts())
print(data["diff_srv_rate"].value_counts())
print(data["srv_diff_host_rate"].value_counts())
print(data["dst_host_count"].value_counts())
print(data["dst_host_srv_count"].value_counts())
print(data["dst_host_same_srv_rate"].value_counts())
print(data["dst_host_diff_srv_rate"].value_counts())
print(data["dst_host_same_src_port_rate"].value_counts())
print(data["dst_host_srv_diff_host_rate"].value_counts())
print(data["dst_host_serror_rate"].value_counts())
print(data["dst_host_srv_serror_rate"].value_counts())
print(data["dst_host_rerror_rate"].value_counts())
print(data["dst_host_srv_rerror_rate"].value_counts())
print(data["label"].value_counts())



df = pd.read_csv('./KDDTrain+.csv')

print(df["label"].value_counts())

# Dropping columns which has correlation with target less than threshold
#target = "label"
#correlations = df.corr()[target].abs()
#correlations = correlations.round(2)
#correlations.to_csv('./lab_f_corr.csv',index=False)
#df=df.drop(correlations[correlations<0.06].index, axis=1)

#print (df.shape,df.columns)
#df.to_csv('./labCorr_out.csv',index=False)


# Dropping process parameter
df1 = df.drop(["label"], axis=1) 
print (df1.shape,df1.columns)

#finding correlation between manipulated & disturbance variables
correlations = df1.corr()
correlations = correlations.round(2)
correlations.to_csv('./MV_DV_corr_Mat_nsl1.csv',index=False)
fig = plt.figure()
g = fig.add_subplot(111)
cax = g.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,40,4)
g.set_xticks(ticks)
g.set_yticks(ticks)
g.set_xticklabels(list(df1.columns))
g.set_yticklabels(list(df1.columns))
plt.savefig('./MV_DV_corr_Mat_nsl1.jpeg')

plt.show()

#removing parameters with high correlation 
upper = correlations.where(numpy.triu(numpy.ones(correlations.shape), k=1).astype(numpy.bool))
cols_to_drop = []
print(upper)
for i in upper.columns:
	if (any(upper[i] == -1) or any(upper[i] == -0.98) or any(upper[i] == -0.99) or any(upper[i] == 0.98) or any(upper[i] == 0.99) or any(upper[i] == 1)):
		cols_to_drop.append(i)
df = df.drop(cols_to_drop, axis=1) 

print (df.shape,df.columns)
df.to_csv('./nsl.csv',index=False)

train_df = pd.read_csv('./nsl.csv')

train_df['protocol_type'] = train_df['protocol_type'].astype('category')

train_df['protocol_type'] = train_df['protocol_type'].cat.codes

train_df['service'] = train_df['service'].astype('category')

train_df['service'] = train_df['service'].cat.codes

train_df['flag'] = train_df['flag'].astype('category')

train_df['flag'] = train_df['flag'].cat.codes

train_df['src_bytes'] = train_df['src_bytes'].astype('category')

train_df['src_bytes'] = train_df['src_bytes'].cat.codes

train_df['dst_bytes'] = train_df['dst_bytes'].astype('category')

train_df['dst_bytes'] = train_df['dst_bytes'].cat.codes

train_df = train_df.drop(columns=['land','wrong_fragment','urgent','hot','num_failed_logins','num_compromised','root_shell','su_attempted','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login'])

train_df.to_csv('check-nsl.csv' , index=False)

print(train_df.head())

#create a dataframe with all training data except the target column
train_X = train_df.drop(columns=['label'])

##check if the target variable has been removed
print(train_X.head())

#one hot encode target column
train_y = to_categorical(train_df.label)

#create model
model = Sequential()

#get the number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(2, activation='softmax'))

#compile the model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#set early stopping monitor,so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=2)

#train the model & fit the model
history=model.fit(train_X, train_y, epochs=25, validation_split=0.2, callbacks=[early_stopping_monitor])

#evaluate the model
loss, accuracy = model.evaluate(train_X, train_y, verbose=True)

print("Accuracy : ",accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

df=pd.read_csv("check-nsl.csv")
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

#training an testing split of the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)

#importing random forest classifier with its arguments
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10000,max_depth=5,
                            random_state=0).fit(X_train, y_train)

y_score_rf = rf.predict_proba(X_test)[:,-1]


predicted = rf.predict(X_test)

print("Accuracy for RFC:",accuracy_score(y_test, predicted))

from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, predicted)  

print(cm)  

#prediction of precision,accuracy,recall scores
from sklearn.metrics import average_precision_score, precision_recall_curve

average_precision = average_precision_score(y_test, predicted)

print('Average precision-recall score RF: {}'.format(average_precision))

precision, recall, _ = precision_recall_curve(y_test, predicted)

print("Precision : ",precision)
print("Recall : ",recall)

#importing gaussian naive bayes clasiifier with its arguments
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Accuracy
print('Accuracy Score for gnb: ', accuracy_score(y_true=y_test, y_pred=gnb.predict(X_test)))

from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  

print(cm)  

average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score GNB: {}'.format(average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_pred)

print("Precision : ",precision)
print("Recall : ",recall)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

# Accuracy
print('Accuracy Score for LR: ', accuracy_score(y_true=y_test, y_pred=logreg.predict(X_test)))
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  

print(cm)  

average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score LRC: {}'.format(average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_pred)

print("Precision : ",precision)
print("Recall : ",recall)

