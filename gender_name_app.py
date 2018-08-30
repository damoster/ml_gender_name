from flask import Flask, request
import ml_module

# TODO make return an http response, i.e. JSON object
# TODO add data validation for POST requests

app = Flask(__name__)
gender_name_model = ml_module.gender_name()
 
@app.route("/retrain")
def retrain():
  accuracy = gender_name_model.retrain()
  return "Model re-training complete! Accuracy: {}%".format(accuracy)

@app.route("/reset")
def reset():
  accuracy = gender_name_model.reset()
  return "Model reset complete! Accuracy: {}%".format(accuracy)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
  if request.method == 'GET':
    name = request.args['name']
    prediction = gender_name_model.predict(name)
  else:
    return "Error with request"

  return "Name: {} Prediction: {}".format(name, prediction)

# Post requests should be json format
# curl -d '{"name":"ray","label":"M"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/add
@app.route("/add", methods=['GET', 'POST'])
def add():
  if request.method == 'POST':
    # print(request.is_json)
    # TODO: add if for above
    request_json = request.get_json()
    name = request_json.get('name')
    label = request_json.get('label')
    
    gender_name_model.add(name, label) # TODO, maybe return new total data count?
  else:
    return "Error with request"

  return "The following data was added to data pool: {} ({})\n".format(name.title(), label)

if __name__ == "__main__":
  # initialise ML model with current data pool
  # currently done in global variable for model definition
  app.run()