import json
import requests

# Send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

# Print the status code
print("GET Status Code:", r.status_code)

# Print the welcome message
print("Welcome Message:", r.text)

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST using the data above
r = requests.post("http://127.0.0.1:8000/predict", json=data)

# Print the status code
print("POST Status Code:", r.status_code)

# Check if the response is successful and has content
if r.status_code == 200 and r.content:
    try:
        # Parse the JSON response
        result = r.json()
        print("POST Result:", result)
    except json.decoder.JSONDecodeError:
        print("Error: Response is not in JSON format")
else:
    print("Error: Failed to get a valid response")
