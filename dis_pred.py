"""
Created on Sat Feb 16 09:35:16 2019

@author: Paras
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
mscaler = MinMaxScaler()
scaler = StandardScaler()

train = pd.read_csv('Train.csv').dropna(how='all').fillna(0)
test = pd.read_csv('test.csv')
y = train.iloc[:, 3:].dropna(how='all')
data = pd.concat([train.iloc[:,:3], test], axis=0)

basket = pd.read_csv('Product_sales_train_and_test.csv')
sample = pd.read_csv('Sample_Submission.csv')
"""details = pd.read_csv('product_details.csv', encoding = "ISO-8859-1").fillna("")
details['pack'] = details['Pack'] + details['Unnamed: 3']
details.drop(['Pack','Unnamed: 3'], axis=1, inplace=True)
details['pack'] = details['pack'].replace("10'S", "10 S")
details['pack'].value_counts()"""

train['Customer'].nunique()
test['Customer'].nunique()

cust = set(train['Customer'])
cust_test = set(test['Customer'])
len(cust_test - cust)

def view_counts(train):
    for frame, group in train.groupby('Customer'):
        print(frame, 'present ', len(group), 'times')
        print("5% : ", (group['5'] == 1).sum())
        print("12% : ", (group['12'] == 1).sum())
        print("18% : ", (group['18'] == 1).sum())
        print("28% : ", (group['28'] == 1).sum())

view_counts(train)

train['5'].value_counts()
train['12'].value_counts()
train['18'].value_counts()
train['28'].value_counts()


### Analysis starts here
baskets = []
for i in basket['Customer_Basket'].values:
    i = i.strip('][')
    l = pd.to_numeric(i.split(' '))
    baskets.append(l)
    
basket['Baskets'] = baskets
basket = basket.drop('Customer_Basket', axis=1)


### Get the counts
tmp = pd.DataFrame()
tmp['BillNo'] = data['BillNo']
tmp = pd.merge(tmp, basket, how='inner', on='BillNo')
counts = []
for i, row in tmp.iterrows():
    c = len(row['Baskets'])
    counts.append(c)
########################################
X = pd.DataFrame(columns = ['BillNo'])
X['BillNo'] = data['BillNo']
X = pd.merge(X, basket, how='inner', on='BillNo')
x = pd.DataFrame(columns=[str(i) for i in range(1001, 1810)])
X = pd.concat([X, x], axis=1)

def get_products(X):
    for i, row in X.iterrows():
        for value in row['Baskets']:
            if row[str(value)] == 1:
                row[str(value)] += 1
            else:
                row[str(value)] = 1
    return X

X = get_products(X)
X = X.fillna(0)
X.drop('Baskets', axis=1, inplace=True)
X['counts'] = counts

### Prepare y
label=[]
y.columns = ['5','12','18','28']
for i, row in y.iterrows():
    if row['5'] == 1:
        label.append(1)
    elif row['12'] == 1:
        label.append(2)
    elif row['18'] == 1:
        label.append(3)
    elif row['28'] == 1:
        label.append(4)
    else:
        label.append(0)
        
y['label'] = label
y.drop(['5','12','18','28'], axis=1, inplace=True)
y = pd.Series(y['label'].values)

### Encoding dates and customers
dates = pd.get_dummies(data['Date'])
dates = dates.reset_index(drop=True)
cust = pd.get_dummies(data['Customer'])
cust = cust.reset_index(drop=True)
X_tmp = pd.concat([X, dates, cust], axis=1)


X_train, X_test = X_tmp.iloc[:12200, 1:], X_tmp.iloc[12200:, 1:]

"""X_t = X_tmp.drop('counts', axis=1)
X_train, X_test = X_t.iloc[:12200, 1:], X_t.iloc[12200:, 1:]"""


clf = LR()

def prob_to_labels(dec):
    dec_pred = []
    for i, row in dec.iterrows():
        l = [0,0,0,0,0]
        index = np.argmax(row)
        l[index] = 1
        dec_pred.append(l)
    dec_pred = pd.DataFrame(dec_pred, columns = [1,2,3,4])
    return dec_pred

def cross_validate(clf, X_train, y, cv=3):
    prob_scores, scores = 0, 0
    for j in range(cv):
        train, test, y_train, y_test = tts(X_train, y, test_size=0.2, 
                                           random_state=j)
    
    
        clf.fit(train, y_train)
        
        ##### Decision scores
        dec = clf.decision_function(test)
        dec = get_prob_scores(dec)
        dec_pred = prob_to_label(dec)
        
        # Verify
        y_dec = pd.get_dummies(y_test)
        #print("\n", j, " iteration")
        print("log loss: ", log_loss(y_dec, dec))
        print("predicted log loss: ", log_loss(y_dec, dec_pred))
        a = accuracy_score(pd.get_dummies(y_test), dec_pred)
        #print("Accuracy using prob scores: ", a)
        
        ##### Straight up predictions
        pred = clf.predict(test)
        s = accuracy_score(y_test, pred)
        #print("Normal predictions: ", s)
        prob_scores += a
        scores += s
        
    print("final prob scores mean: ", prob_scores/cv)
    print("final scores mean: ", scores/cv)

cross_validate(X_train, y, 100)

### Fit on whole data
clf.fit(X_train, y)

### Get probability scores
def get_prob_scores(dec):
    normalizedArray = []
    for row in range(0, len(dec)):
        l = []
        Min =  min(dec[row])
        Max = max(dec[row])
        for element in dec[row]:
            l.append(float(element-Min)/float(Max- Min) )
        normalizedArray.append(l)
        
    #Normalize to 1
    newArray = []
    for row in range(0, len(normalizedArray)):
        li = [x / sum(normalizedArray[row]) for x in normalizedArray[row]]
        newArray.append(li)
        
    sample_p = pd.DataFrame(newArray, columns=[1,2,3,4])
    return sample_p

### Manipulating test
sample_pred = clf.decision_function(X_test)[:,1:]
sample_prob = get_prob_scores(sample_pred)
sample_prob.to_csv('with_dates_actual_2_prob.csv', index=False)
sample_p = prob_to_labels(sample_prob)

sample_pred = clf.predict(X_test)
sample_p = pd.get_dummies(sample_pred)

sample_p.to_csv('with_dates_2.csv', index=False)

predictions = pd.read_csv('with_dates_1.csv')
prob = pd.read_csv('with_dates_1_prob.csv')

actual = []
for i, row in prob.iterrows():
    r = row[1:].sort_values(ascending=False)
    maxx = r[0]
    second_max = r[1]
    l = [0,0,0,0]
    if maxx - second_max > 0.84:
        for j in range(1,5):
            if row[j] == maxx:
                l[j-1] = 1
                break
    actual.append(l)

actual = []
for i, row in prob.iterrows():
    row = row.sort_values(ascending=False)
    l = [0,0,0,0,0]
    if row[0] - row[1] >= 0.3:
        index = int(np.argmax(row))
        l[index] = 1
    actual.append(l)

sample_df = pd.DataFrame(actual, columns=[1,2,3,4])
sample['Discount 5%'] = sample_df[1]
sample['Discount 12%'] = sample_df[2]
sample['Discount 18%'] = sample_df[3]
sample['Discount 28%'] = sample_df[4]
sample.to_csv('modified_84.csv', index=False)


sample['Discount 5%'] = sample_prob[1]
sample['Discount 12%'] = sample_prob[2]
sample['Discount 18%'] = sample_prob[3]
sample['Discount 28%'] = sample_prob[4]

sample.to_csv('modified_new_prob.csv', index=False)

### Check counts
one = pd.read_csv('modified_80.csv')
two = pd.read_csv('modified_84.csv')

def check_result(one):
    s = 0
    for col in one.columns[1:]:
        print(len(one) - one[col].value_counts()[0])
        s += len(one) - one[col].value_counts()[0]
    print("Count: ", s, '/', len(one))
    
check_result(one)
check_result(two)

def get_predictions(per, sample_pred_prob):
    actual = []
    for i, row in sample_pred_prob.iterrows():
        r = list(row.sort_values(ascending=False))
        maxx = r[0]
        second_max = r[1]
        l = [0,0,0,0]
        if maxx - second_max > per:
            for j in range(1,5):
                if row[j] == maxx:
                    l[j-1] = 1
                    break
        actual.append(l)
    return actual

for i in [10.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8]:
    one = pd.DataFrame(get_predictions(i/100.00, prob), columns=[1,2,3,4])
    check_result(one)
    print(i, ":")

