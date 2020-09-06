## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Flask (for API) installed.

### Project Structure
This project has four major parts :
1. index.html - This contains code for accepting xray images to detect pneumonia
2. app.py - This contains Flask APIs that receives images from GUI or API calls, computes the precited value based on our model and returns it.
3. model.h5 - pretrained model

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000
# flask_deployment_pneumonia_detection GL Capstone Project
