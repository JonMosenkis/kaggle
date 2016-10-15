# encoding=utf-8

from csv import DictReader
import numpy
from sklearn import preprocessing


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
    return processed
