# encoding=utf-8

from csv import DictReader
import numpy
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plot


def parse_data():
    data, survived, sex = [], [], []
    with open('train.csv') as infile:
        raw_data = DictReader(infile, delimiter=',', quotechar='"')
        for row in raw_data:
            if row['Sex'] == 'male':
                sex.append(1)
            elif row['Sex'] == 'female':
                sex.append(0)
            else:
                raise AssertionError('Sex not male or female')

            data.append([float(x) for x in [row['Pclass'], row['SibSp'], row['Parch'], row['Fare']]])
            survived.append(row['Survived'])

    data, sex = numpy.array(data), numpy.array(sex)
    processed = numpy.column_stack((preprocessing.normalize(data, norm='max'), sex))
    return {
        'classifiers': processed,
        'labels': survived
    }


def test_logistic_regression(X, y):
    learner = LogisticRegression(penalty='l1')
    train_sizes, train_scores, test_scores = learning_curve(learner, X, y)
    plot.figure()
    plot.xlabel('Training Examples')
    plot.ylabel('Score')
    plot.grid()
    plot.plot(train_sizes, numpy.mean(train_scores, axis=1), color='r', label='training scores')
    plot.plot(train_sizes, numpy.mean(test_scores, axis=1), color='g', label='test_scores')
    return plot

data = parse_data()
test_logistic_regression(data['classifiers'], data['labels'])
plot.show()
