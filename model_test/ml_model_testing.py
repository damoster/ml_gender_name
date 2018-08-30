import random
import nltk
import csv

# Relevant gender features of a name
def gender_features(name):
  return {
    'prefix1':name[:1],
    'suffix1':name[-1:],
    'suffix2':name[-2:],
    'suffix3':name[-3:],
    'length':len(name)
  }

# Import data
labelled_names = []
countM = 0
countF = 0
with open('../data/name_gender.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for row in csv_reader:
    labelled_names.append((row[0], row[1]))
    if row[1] == 'M':
      countM += 1
    elif row[1] == 'F':
      countF += 1
    else:
      print("ERROR"+row[1])
      exit()

print("M : {}".format(countM))
print("F : {}".format(countF))

random.shuffle(labelled_names)

# Remove class bias, but randomly removing female names until equal number to males
# Using this just to test features/parameters etc. in prod, may recieve more F names
names = []
count = 0
for row in labelled_names:
  if row[1] == 'F' and count < countM:
    count += 1
    names.append(row)
  elif row[1] == 'M':
    names.append(row) 

print(len(names)/2)
num_train = int(0.66 * len(names))
num_test = len(names) - num_train

featuresets = [(gender_features(n), gender) for (n, gender) in names]
train_set, test_set = featuresets[num_train:], featuresets[:num_test]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Testing classifier
print(classifier.classify(gender_features('Neo')))
print(classifier.classify(gender_features('Trinity')))

print(nltk.classify.accuracy(classifier, test_set)) # check accuracy



