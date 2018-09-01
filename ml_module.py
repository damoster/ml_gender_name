import random
import nltk
import csv
from shutil import copyfile

class gender_name:
  def __init__(self):
    self.init_csv = './data/init_data.csv'
    self.curr_csv = './data/curr_data.csv'
    self.prop_train = 0.66 # % proportion of data to use for training set
    self.classifier = None

    self.retrain() # Re-build the model on initialisation

  # Relevant gender features of a name
  def gender_features(self, name):
    return {
      'prefix1':name[:1],
      'suffix1':name[-1:],
      'suffix2':name[-2:],
      'suffix3':name[-3:],
      'suffix4':name[-4:],
      'suffix5':name[-5:],
      'length':len(name)
    }

  # Retrain the model using current data pool
  def retrain(self):
    labelled_names = []
    with open(self.curr_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        labelled_names.append((row[0], row[1]))

    random.shuffle(labelled_names)
    names = labelled_names
    num_train = int(self.prop_train * len(names))
    num_test = len(names) - num_train

    featuresets = [(self.gender_features(n), gender) for (n, gender) in names]
    train_set, test_set = featuresets[num_train:], featuresets[:num_test]
    self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    return(nltk.classify.accuracy(self.classifier, test_set))

  # Reset current data pool to the initial data set
  def reset(self):
    copyfile(self.init_csv, self.curr_csv)
    return(self.retrain())

  # Predict the gender of the given input name
  def predict(self, name):
    return self.classifier.classify(self.gender_features(name))
    
  # Add a labelled data entry to the current data pool
  # @pre-condition: name and label are strings
  def add(self, name, label):
    label = label.upper()
    if not (label == 'M' or label == 'F'):
      return (False, "Invalid label, label must be 'M' or 'F'")

    # Check if name already exists or not
    with open(self.curr_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        if name.lower() == row[0].lower():
          return (False, "Name already exists in data pool")

    try:
      with open(self.curr_csv,'a') as fd:
        fd.write(name.title()+","+label+"\n")
      return (True, "Name was successfully added to data pool")
    except:
      return (False, "Name was not able to be successfully added to data pool")
