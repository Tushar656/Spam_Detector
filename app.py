from flask import Flask, request, render_template
from spam import spamModel


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('spamDetect.html', isSpam="")


@app.route('/check', methods=["POST", "GET"])
def chk():
    message = request.form['inpText']
    arr = ['Not Spam', "Spam"]
    return render_template('spamDetect.html', isSpam=arr[spamModel(message)[0]])

