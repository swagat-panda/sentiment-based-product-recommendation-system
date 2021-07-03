from flask import Flask, render_template, request

from load_model import LoadModel
from process_output import FormartOutput

app = Flask(__name__)
model = LoadModel()
obj = FormartOutput()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    user_response = userText.lower()
    try:
        user_response=int(user_response)
    except Exception as e:
        return "Please enter a uid that is an Integer."

    try:
        response =obj.process(user_response)
    except KeyError as e:
        return "This uid is not present in our Training data.Can you please try something else."
    except Exception as e:
        print(e)
        return "Sorry,something went Wrong,can you try again."
    return response


if __name__ == "__main__":
    app.run(port=8000)
