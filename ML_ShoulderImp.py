
#Classification


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import time

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, VotingClassifier)
from sklearn.naive_bayes import (GaussianNB, BernoulliNB)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report

    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print('tf version {}'.format(tf.__version__))
if tf.test.is_gpu_available():
    print(tf.test.gpu_device_name())
else:
    print('TF cannot find GPU')

X_train = np.load('tr_f_Dense201.npy')
y_train = np.load('tr_l_Dense201.npy')

X_test = np.load('tst_f_Dense201.npy')
y_test = np.load('tst_l_Dense201.npy')


ny_test = np.empty(60)

for i in range(y_test.shape[0]): 
    for j in range(y_test.shape[1]):
        if y_test[i][j] == 1:
            ny_test[i] = j
            
ny_train = np.empty(537)

for i in range(y_train.shape[0]): 
    for j in range(y_train.shape[1]):
        if y_train[i][j] == 1:
            ny_train[i] = j


classifiers = [
        
    ('LR', LogisticRegression(random_state=0, C = 10, multi_class = 'multinomial', solver= 'newton-cg'))
    ('LSVM', LinearSVC(random_state=0, C = 0.1, multi_class = 'crammer_singer')),
    ('KNN', KNeighborsClassifier(n_neighbors=1, p=2)),
    ('DT', DecisionTreeClassifier(criterion='entropy', max_depth=58)),
    ('RF', RandomForestClassifier(criterion='gini', max_depth=400)),
    ('AB', AdaBoostClassifier(n_estimators=50, learning_rate=1.0)),
    ('GNB', GaussianNB()),
    ('BNB', BernoulliNB()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('MLP', MLPClassifier())

    ]

for clf_name, clf in classifiers:
    
    cross_val_scoree = np.empty(5)
   
    clf = make_pipeline(StandardScaler(with_mean = True, with_std = True), clf)
    clf.fit(X_train, ny_train)
    
    cv = ShuffleSplit(n_splits=5, random_state=0)
    
    cross_val_scoree = cross_val_score(clf, X_train, ny_train, cv=cv)
    
    print('Classifier: %s' % clf_name)
    print(cross_val_scoree)
    print('%0.4f accuracy with a standard deviation of %0.4f' % (np.mean(cross_val_scoree), np.std(cross_val_scoree)))
   
    
       
    train_sizes, train_scores, test_scores, fit_times, _  = learning_curve(estimator = clf,
                                                            X = X_train,
                                                            y = ny_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 5),
                                                            cv = cv,
                                                            return_times = True,)
    
    
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
       
    plt.plot(train_sizes, train_mean, color='red', 
         marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='red')
    
    
    plt.plot(train_sizes, test_mean, color='green', linestyle='--',
             marker='s', markersize=5, label='Cross-validation accuracy')
    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.title('Learning Curve of the {0}'.format(clf_name))
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.1, 1.05])
    plt.savefig('Learning_Cur/{0}_LearningCurve_DenseML_ShoulderImp.png'.format(clf_name), dpi = 600)
    plt.show()
     

    plt.plot(fit_times_mean, test_mean, 'o-')
    plt.fill_between(fit_times_mean, test_mean - test_std,
                          test_mean + test_std, alpha=0.1)
    
    plt.grid()
    plt.title('Performance of the {0}'.format(clf_name))
    plt.xlabel('Fit times')
    plt.ylabel('Accuracy')
    plt.savefig('Time_Perf/{0}_TimePerformance_DenseML_ShoulderImp.png'.format(clf_name), dpi = 600)
    plt.show()
    
    data = {'train_sizes':train_sizes,
            'train_mean':train_mean,
            'test_mean':test_mean,
            'fit_times_mean':fit_times_mean}

    df = pd.DataFrame(data)
    df.to_csv('Scores/{0}_TrainingScores_DenseML_ShoulderImp.csv'.format(clf_name), index = False)

     
    t0 = time.time()
    prediction = clf.predict(X_test)
    t1 = time.time()
    testing_time = t1 - t0
    print('Testing time: %0.4fs' % testing_time)
    print(classification_report(ny_test, prediction, digits=4))
    print('*'*50,'\n')
    score = round(accuracy_score(ny_test, prediction),4) 
    cm1 = cm(ny_test, prediction)
    sns.heatmap(cm1, cmap='YlOrRd', annot=True, annot_kws={'size': 16}) 
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Accuracy Score: {0}'.format(score), size = 12)
    plt.savefig('Conf_Mtx/{0}_ConfMtx_DenseML_ShoulderImp.png'.format(clf_name), dpi = 600)
    plt.show()
    
    
    """
    cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    ('LR', LogisticRegression(random_state=0, C = 10, multi_class = 'multinomial', solver= 'newton-cg'))
    ('LSVM', LinearSVC(random_state=0, C = 0.1, multi_class = 'crammer_singer')),
    ('KNN', KNeighborsClassifier(n_neighbors=1, p=2)),
    ('DT', DecisionTreeClassifier(criterion='entropy', max_depth=58)),
    ('RF', RandomForestClassifier(criterion='gini', max_depth=400)),
    ('AB', AdaBoostClassifier(n_estimators=50, learning_rate=1.0)),
    ('GNB', GaussianNB()),
    ('BNB', BernoulliNB()),
    
    estimator = [
        
    ('LR', LogisticRegression(random_state=0, C = 10, multi_class = 'multinomial', solver= 'newton-cg')),
    ('KNN', KNeighborsClassifier(n_neighbors=1, p=2)),
    ('DT', DecisionTreeClassifier(criterion='entropy', max_depth=58)),
    ('RF', RandomForestClassifier(criterion='gini', max_depth=400)),
    ('AB', AdaBoostClassifier(n_estimators=50, learning_rate=1.0)),
    ('GNB', GaussianNB()),
    ('BNB', BernoulliNB()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('MLP', MLPClassifier())
          
    ]
classifiers = [
        
    ('VCH', VotingClassifier(estimators = estimator, voting ='hard')),
    ('VCS', VotingClassifier(estimators = estimator, voting ='soft'))

    ]
    """