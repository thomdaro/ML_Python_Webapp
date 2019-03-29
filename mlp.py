import csv
from sklearn.neural_network import MLPClassifier
import random

with open('cars_data.txt') as f:
    data = list(csv.reader(f))

prices = ['vhigh', 'high', 'med', 'low'] # used to convert strings to ints via index()
maintain = ['vhigh', 'high', 'med', 'low']
luggage = ['small', 'med', 'big']
safety = ['low', 'med', 'high']
classes = ['unacc', 'acc', 'good', 'vgood']

for point in data:
    point[0] = prices.index(point[0])
    point[1] = maintain.index(point[1])
    if point[2] == '5more': # changes '5more' to just 5 to stay consistent with other int values
        point[2] = 5
    else:
        point[2] = int(point[2])
    if point[3] == 'more': # same as above
        point[3] = 5
    else:
        point[3] = int(point[3])
    point[4] = luggage.index(point[4])
    point[5] = safety.index(point[5])
    point[6] = classes.index(point[6])

model = MLPClassifier(max_iter=1000) # generic classifier with max_iter cranked up (testing showed 90-92% accuracy)
training_data = random.sample(data, 1380) # sample 80% of data
features = [point[:6] for point in training_data]
classes = [point[6] for point in training_data]

model.fit(features, classes)
