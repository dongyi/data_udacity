#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_stock_value', 'from_this_person_to_poi', 'total_payments', 
'to_messages', 'bonus', 'from_poi_to_this_person', 'long_term_incentive', 'deferred_income', 
'restricted_stock_deferred', 'deferral_payments', 'expenses', 'director_fees', 'loan_advances', 
'salary', 'other', 'restricted_stock', 'shared_receipt_with_poi', 'from_messages', 'exercised_stock_options']

### Load the Dictionary Containing the Dataset
all_features = list(data_dict.values())[0].keys()

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

# remove total:
del data_dict['TOTAL']
import pandas as pd
import numpy as np


pl = []
for k, v in data_dict.items():
    l = v
    l.update({'name': k})
    pl.append(l)
df = pd.DataFrame(pl)

# look into data:
print(df.describe().T)



### remove NAN's from dataset, show top 5 salary
top_salary = df[df['salary'] != 'NaN'].sort_values('salary', ascending=False)['name'][:5]

for name in top_salary:
    del data_dict[name]


    
### plot features
import seaborn
data = featureFormat(data_dict, ["salary", "bonus"])

for point in data:
    plt.scatter( point[0], point[1] )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### new features: fraction_to_poi_email,fraction_from_poi_email
df.loc[df['from_poi_to_this_person'] == 'NaN', 'from_poi_to_this_person'] = 0.0
df.loc[df['from_this_person_to_poi'] == 'NaN', 'from_this_person_to_poi'] = 0.0
df.loc[df['to_messages'] == 'NaN', 'to_messages'] = -1
df.loc[df['from_messages'] == 'NaN', 'from_messages'] = -1

df['fraction_from_poi_email'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi_email'] = df['from_this_person_to_poi'] / df['from_messages']

name_to_fraction_from_poi_email = zip(df['name'], df['fraction_from_poi_email'])
name_to_fraction_to_poi_email = zip(df['name'], df['fraction_to_poi_email'])

data_dict.update(dict(name_to_fraction_from_poi_email))
data_dict.update(dict(name_to_fraction_to_poi_email))

my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn.cross_validation import KFold

for train_indices, test_indices in KFold(len(labels), 3):
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier
import time

t0 = time.time()

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print('accuracy before tuning ', score)

print('consume time(Decision tree):', round(time.time() - t0, 3), 's')


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

### try Naive Bayes for prediction
#t0 = time.time()

#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#accuracy = accuracy_score(pred,labels_test)
#print(accuracy)

#print("consume time (naive bayes):", round(time.time() - t0, 3), "s")



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
