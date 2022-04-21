import numpy as np
import pandas as pd
data = pd.read_csv("CreditCard.csv")
data
data.shape
data.info()
data.isnull().sum()
data.describe()
data['Class'].value_counts()
fraud = data.loc[data['Class'] == 1]
len(fraud.value_counts())
normal = data.loc[data['Class'] == 0]
len(normal.value_counts())
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(data['Time'],data['Amount'])
plt.title('Time v/s Amount Graph')
plt.ylabel = 'Amount'
plt.xlabel = 'Time'
plt.show()
plt.scatter(data['Time'],data['Amount'])
plt.title('Time v/s Amount Graph')
plt.ylabel = 'Amount'
plt.xlabel = 'Time'
plt.show()
x = data.iloc[:,data.columns != 'Class']
y = data['Class']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
clf_1 = LogisticRegression(solver = 'liblinear')
clf_1.fit(x_train, y_train)
y_pred = np.array(clf_1.predict(x_test))
y = np.array(y_test)
clf_1_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_1_accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf_2 = LinearDiscriminantAnalysis()
clf_2.fit(x_train, y_train)
y_pred = np.array(clf_2.predict(x_test))
y = np.array(y_test)
clf_2_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_2_accuracy
from sklearn.naive_bayes import GaussianNB
clf_3 = GaussianNB()
clf_3.fit(x_train, y_train)
y_pred = np.array(clf_3.predict(x_test))
y = np.array(y_test)
clf_3_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_3_accuracy
from sklearn.tree import DecisionTreeClassifier
clf_4 = DecisionTreeClassifier()
clf_4.fit(x_train, y_train)
y_pred = np.array(clf_4.predict(x_test))
y = np.array(y_test)
clf_4_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_4_accuracy
from sklearn.ensemble import RandomForestClassifier
clf_5 = RandomForestClassifier()
clf_5.fit(x_train, y_train)
y_pred = np.array(clf_5.predict(x_test))
y = np.array(y_test)
clf_5_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_5_accuracy
from sklearn.svm import SVC
clf_6 = SVC()
clf_6.fit(x_train, y_train)
y_pred = np.array(clf_6.predict(x_test))
y = np.array(y_test)
clf_6_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_6_accuracy
from sklearn.neighbors import KNeighborsClassifier
clf_7 = KNeighborsClassifier()
clf_7.fit(x_train, y_train)
y_pred = np.array(clf_7.predict(x_test))
y = np.array(y_test)
clf_7_accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
clf_7_accuracy
model_df = pd.DataFrame({'Model' : ['LogisticRegression', 'LinearDiscriminantAnalysis', 'GaussianNB', 'DecisionTreeClassifier',
                                    'RandomForestClassifier', 'SVC', 'KNeighborsClassifier'],
                         'Score' : [clf_1_accuracy, clf_2_accuracy, clf_3_accuracy, clf_4_accuracy, clf_5_accuracy,
                                    clf_6_accuracy, clf_7_accuracy]})
model_df.sort_values(by = 'Score', ascending = False)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_1, x_test, y_test)
plot_confusion_matrix(clf_2, x_test, y_test)
plot_confusion_matrix(clf_3, x_test, y_test)
plot_confusion_matrix(clf_4, x_test, y_test)
plot_confusion_matrix(clf_5, x_test, y_test)
plot_confusion_matrix(clf_6, x_test, y_test)
plot_confusion_matrix(clf_7, x_test, y_test)
