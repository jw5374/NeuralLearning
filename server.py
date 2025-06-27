from flask import Flask, request

app = Flask(__name__)


@app.get("/")
def hello_world():
    return "Hello, World!"


@app.post("/nn/number")
def processDrawing():
    req = request.get_json()
    submitted_number = req["number"]
    print(submitted_number)
    for row in submitted_number:
        print("".join([" " if p == 0 else "@" for p in row]))
    return "here!", 200
