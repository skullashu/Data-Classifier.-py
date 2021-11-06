# importing dependencies 
import panadas 
from pandas.plotting import scatter_matrix 
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscrimininantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn import model_selection 
from sklearn.ensemble import VotingClassifier 

#loading dataset 
url= "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv" 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
dataset = read_csv(url, names=names)

#checking dimensions
print(dataset.shape)

#peeking at data 
print(dataset.head(40)) 

#statistical summary 
print(dataset.describe()) 

#class distribution of dataset 
print(dataset.groupby('class').size) 

#univariate plots- box and whisker plots 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 

#plotting histogram of dataset 
dataset.hist()
pyplot.show() 

#multivaariate plots 
scatter_matrix(dataset) 
pyplot.show() 

#creating validation dataset 
#splitting dataset 
array = dataset.values 
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation=train_test_split(X, Y, test_size=0.2, random_size=1) 

#Logistic Regression 
#LDA
#K-Nearesr Neighbors 
#Classification and Regression Trees 
#Gaussian Naive Bayes 
#Support Vector Machine 
#Buiding models 
model=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('NB', GaussianNB())) 

#results 
results=[] 
names=[] 
for name, model in models: 
  kfold = StratifiedKFold(n_splits=10, random_state=1) 
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') 
 
results.append(cv_results) 
names.append(name) 
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) 

#comparing models 
pyplot.boxplot(results, labels=names) 
pyplot.title('Algorithm Comparison') 
pyplot.show() 

#prediction model on svm 
model = SVC(gamma='auto') 
model.fit(X_train, Y_train) 
predictions = model.predict(X_validation) 

#calculations of predictions 
print(accuracy_score(Y_validation, predictions)) 
print(confusion_matrix(Y_validation, predictions)) 
print(classification_report(Y_validation, predictions))

models appned(('SVM', SVC(gamma='auto')))

