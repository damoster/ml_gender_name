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
    male_names = []
    female_names = []
    with open(self.curr_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        if row[1] == 'M':
          male_names.append((row[0], row[1]))
        elif row[1] == 'F':
          female_names.append((row[0], row[1]))
        else:
          raise ValueError('Invalid label found in CSV: {} {}'.format(row[0], row[1]))
  
    random.shuffle(male_names)
    random.shuffle(female_names)

    # Adjusting train set to ensure no gender bias by moving difference into test set
    test_names = []
    if len(male_names) > len(female_names):
      diff = len(male_names) - len(female_names)
      test_names = male_names[:diff]
      
      male_names = male_names[diff:]
    elif len(male_names) < len(female_names):
      diff = len(female_names) - len(male_names)
      test_names = female_names[:diff]

      female_names = female_names[diff:]

    # Num male and female will be equal after above adjustment
    num_train = int(self.prop_train * len(male_names))
    train_names = male_names[:num_train] + female_names[:num_train]
    test_names = test_names + male_names[num_train:] + female_names[num_train:] 

    train_set = [(self.gender_features(n), gender) for (n, gender) in train_names]
    test_set = [(self.gender_features(n), gender) for (n, gender) in test_names]

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
