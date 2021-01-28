from flask import Flask, render_template, request
import pickle

# Load the countvectorizer model from the disk
filename1 = r'C:\Users\admin\Desktop\Movie_genre_prediction_cv-transform.pkl'
cv = pickle.load(open(filename1, 'rb'))

# Load the nb_classifier model from the disk
filename2 = r'C:\Users\admin\Desktop\movie-genre-mnb-model.pkl'
classifier = pickle.load(open(filename2, 'rb'))

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        message = request.form['message']
        data = ['message']
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)