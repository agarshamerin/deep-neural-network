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

data=pd.read_csv("UNSW_NB15_training-set.csv");
nullData = data.isnull().sum()

print(nullData)

print(data["id"].value_counts());
print(data["dur"].value_counts());
print(data["proto"].value_counts());
print(data["service"].value_counts());
print(data["state"].value_counts());
print(data["spkts"].value_counts());
print(data["dpkts"].value_counts());
print(data["sbytes"].value_counts());
print(data["dbytes"].value_counts());
print(data["rate"].value_counts());
print(data["sttl"].value_counts());
print(data["dttl"].value_counts());
print(data["sload"].value_counts());
print(data["dload"].value_counts());
print(data["sloss"].value_counts());
print(data["dloss"].value_counts());
print(data["sinpkt"].value_counts());
print(data["dinpkt"].value_counts());
print(data["sjit"].value_counts());
print(data["djit"].value_counts());
print(data["swin"].value_counts());
print(data["stcpb"].value_counts());
print(data["dtcpb"].value_counts());
print(data["dwin"].value_counts());
print(data["tcprtt"].value_counts());
print(data["synack"].value_counts());
print(data["ackdat"].value_counts());
print(data["smean"].value_counts());
print(data["dmean"].value_counts());
print(data["trans_depth"].value_counts());
print(data["response_body_len"].value_counts());
print(data["ct_srv_src"].value_counts());
print(data["ct_state_ttl"].value_counts());
print(data["ct_dst_ltm"].value_counts());
print(data["ct_src_dport_ltm"].value_counts());
print(data["ct_dst_sport_ltm"].value_counts());
print(data["ct_dst_src_ltm"].value_counts());
print(data["is_ftp_login"].value_counts());
print(data["ct_ftp_cmd"].value_counts());
print(data["ct_flw_http_mthd"].value_counts());
print(data["ct_src_ltm"].value_counts());
print(data["ct_srv_dst"].value_counts());
print(data["is_sm_ips_ports"].value_counts());
print(data["attack_cat"].value_counts());
print(data["label"].value_counts());


df = pd.read_csv('./UNSW_NB15_training-set.csv')

print(df["label"].value_counts())

# Dropping columns which has correlation with target less than threshold
#target = "label"
#correlations = df.corr()[target].abs()
#correlations = correlations.round(2)
#correlations.to_csv('./lab_f_corr.csv',index=False)
#df=df.drop(correlations[correlations<0.06].index, axis=1)

#print (df.shape,df.columns)
#df.to_csv('./labCorr_out.csv',index=False)

#correlation based feature selection
# Dropping process parameter
df1 = df.drop(["label"], axis=1) 
print (df1.shape,df1.columns)

#finding correlation between manipulated & disturbance variables
correlations = df1.corr()
correlations = correlations.round(2)
correlations.to_csv('./MV_DV_corr_Mat_unsw-nb.csv',index=False)
fig = plt.figure()
g = fig.add_subplot(111)
cax = g.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,40,4)
g.set_xticks(ticks)
g.set_yticks(ticks)
g.set_xticklabels(list(df1.columns))
g.set_yticklabels(list(df1.columns))
plt.savefig('./MV_DV_corr_Mat_unsw-nb.jpeg')
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
df.to_csv('./unswnb15.csv',index=False)

train_df = pd.read_csv('./unswnb15.csv')

train_df['dur'] = train_df['dur'].astype('category')

train_df['dur'] = train_df['dur'].cat.codes

train_df['proto'] = train_df['proto'].astype('category')

train_df['proto'] = train_df['proto'].cat.codes

train_df['state'] = train_df['state'].astype('category')

train_df['state'] = train_df['state'].cat.codes

train_df['attack_cat'] = train_df['attack_cat'].astype('category')

train_df['attack_cat'] = train_df['attack_cat'].cat.codes

train_df = train_df.drop(columns=['service','dinpkt','djit','stcpb','dtcpb','dwin','tcprtt','synack','ackdat'])

train_df.to_csv('check.csv' , index=False)

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

df=pd.read_csv("check.csv")

X = df[df.columns[:-1]]
y = df[df.columns[-1]]

#training an testing split of the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)

#importing random forest classifier with its arguments
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000,max_depth=5,
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

print('Average precision-recall score RFC: {}'.format(average_precision))

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

print('Average precision-recall score gnb: {}'.format(average_precision))

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


