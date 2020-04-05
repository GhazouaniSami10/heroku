# import the Flask class from the flask module
from ObjectDetector import Detector
import io

from flask import Flask, render_template, request

from PIL import Image
from flask import send_file
# create the application object
app = Flask(__name__)
app._static_folder = "/home/gladiateur/Desktop/flask/corona/templates"


detector = Detector()

@app.route("/")

@app.route('/welcome')
@app.route('/home')

def welcome():
    return render_template('home.html')  # render a template

@app.route('/login')
def login():
    return render_template('login.html')  

@app.route('/register')
def register():
    return render_template('home.html') 

@app.route('/about')
def about():
    return render_template('about.html') 


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()
@app.route("/", methods=['POST'])
def upload():
    if request.method == 'POST':
        file = Image.open(request.files['file'].stream)
        img = detector.detectObject(file)
        return send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')



if __name__ == "__main__":
    app.run()
