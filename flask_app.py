# largely taken from the textbook, modified slightly for my data and classifier

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(cur_dir, 'pkl_objects/model.pkl'), 'rb'))


def classify(point):
    return model.predict(point)[0]


class InputForm(Form):
    priceInput = TextAreaField('', [validators.DataRequired()])
    maintainInput = TextAreaField('', [validators.DataRequired()])
    doorsInput = TextAreaField('', [validators.DataRequired()])
    passengersInput = TextAreaField('', [validators.DataRequired()])
    luggageInput = TextAreaField('', [validators.DataRequired()])
    safetyInput = TextAreaField('', [validators.DataRequired()])


@app.route('/')
def index():
    form = InputForm(request.form)
    return render_template('main.html', form=form)


@app.route('/output', methods=['POST'])
def output():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        price = request.form['priceInput']
        maint = request.form['maintainInput']
        doors = request.form['doorsInput']
        passengers = request.form['passengersInput']
        luggage = request.form['luggageInput']
        safety = request.form['safetyInput']
        # convert strings to ints for classification but cast output to string for display on the page
        out = str(classify([[int(price), int(maint), int(doors), int(passengers), int(luggage), int(safety)]]))
        return render_template('output.html',
                               price=price,
                               maint=maint,
                               doors=doors,
                               passengers=passengers,
                               luggage=luggage,
                               safety=safety,
                               prediction=out)
    return render_template('main.html', form=form)


if __name__ == '__main__':
    app.run()
