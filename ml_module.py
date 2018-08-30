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
    #
    self.retrain() # Re-build the model on initialisation
    #
  # Relevant gender features of a name
  def gender_features(self, name):
    return {
      'prefix1':name[:1],
      'suffix1':name[-1:],
      'suffix2':name[-2:],
      'suffix3':name[-3:],
      'length':len(name)
    }
  def retrain(self):
    # Import data
    labelled_names = []
    # countM = 0
    # countF = 0
    with open(self.curr_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        labelled_names.append((row[0], row[1]))
        # if row[1] == 'M':
        #   countM += 1
        # elif row[1] == 'F':
        #   countF += 1
        # else:
        #   print("ERROR"+row[1])
        #   exit()
    #
    random.shuffle(labelled_names)
    names = labelled_names
    num_train = int(self.prop_train * len(names))
    num_test = len(names) - num_train
    #
    featuresets = [(self.gender_features(n), gender) for (n, gender) in names]
    train_set, test_set = featuresets[num_train:], featuresets[:num_test]
    self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    #
    return(nltk.classify.accuracy(self.classifier, test_set))
    #
  def reset(self):
    copyfile(self.init_csv, self.curr_csv)
    return(self.retrain())    
  def predict(self, name):
    return self.classifier.classify(self.gender_features(name))
    #
  def add(self, name, label):
    # TODO should type check and validation be done in module or in HTTP?
    # I think should be done in HTTP so we assume we have valid input here
    # Should the try catches be here or in HTTP? I think in the HTTP

    # TODO should probs check for duplicates before adding
    with open(self.curr_csv,'a') as fd:
      fd.write(name.title()+","+label+"\n")
