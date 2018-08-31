from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import ml_module

# Useful URLS
# http://blog.luisrei.com/articles/flaskrest.html
# https://restfulapi.net/http-status-codes/

app = Flask(__name__)
CORS(app) # For Swagger UI integration
gender_name_model = ml_module.gender_name()

def errorResponse(error_msg, status_code):
  data = {
      'status':status_code,
      'errorMsg':error_msg
  }
  resp = jsonify(data)
  resp.status_code = status_code
  return resp

@app.route("/retrain", methods=['POST'])
def retrain():
  try:
    accuracy = gender_name_model.retrain()
    data = {
        'accuracy':accuracy
    }
    resp = jsonify(data)
    resp.status_code = 200
    return resp
  except Exception as e:
    # print(e)
    return errorResponse("Internal Server Error",500)

@app.route("/reset", methods=['POST'])
def reset():
  try:
    accuracy = gender_name_model.reset()
    data = {
        'accuracy':accuracy
    }
    resp = jsonify(data)
    resp.status_code = 200
    return resp
  except Exception as e:
    # print(e)
    return errorResponse("Internal Server Error",500)

@app.route("/predict", methods=['GET'])
def predict():
  try:
    name = request.args['name']
  except:
    return errorResponse("Invalid input", 400)

  prediction = gender_name_model.predict(name)
  
  data = {
    'prediction':prediction
  }
  resp = jsonify(data)
  resp.status_code = 200
  return resp

# curl -d '{"name":"ray","label":"M"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/add
@app.route("/add", methods=['POST', 'OPTION'])
def add():
  if request.is_json: # Content-Type: application/json
    request_json = request.get_json()
    try:
      name = request_json.get('name')
      label = request_json.get('label')
    except:
      return errorResponse("Invalid input", 400)
  else:
    return errorResponse("Invalid input", 400)

  is_success = gender_name_model.add(name, label) # TODO, maybe return new total data count?
  data = {
    'success':is_success
  }
  resp = jsonify(data)
  resp.status_code = 200
  return resp

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0')