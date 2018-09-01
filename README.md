# ML name gender server

## Overview
Machine learning application server with endpoints to interact with the namge gender machine learning model and predict genders of a given name. Documentation of the endpoints can be found in the swagger/swagger.yml file.

## Requirements
Python 3.6.5+

## Usage
To run the server, please execute the following from the root directory:

```
pip3 install -r requirements.txt
python3 app.py
```

and open your browser to here:

```
http://127.0.0.1:5000/
```

The API documentation can be found in the following directory:
```
swagger/swagger.yml
```
Import the file onto https://editor.swagger.io/ to view in Swagger Editor

## Running with Docker

To run the server on a Docker container, please execute the following from the root directory:

```bash
# building the image
docker build -t name_gender_server .

# starting up a container
docker run -p 8080:8080 name_gender_server
```

Alternatively, using the provided shell script with your preferred 'image_name'

```bash
sh build_docker.sh <image_name>
```
