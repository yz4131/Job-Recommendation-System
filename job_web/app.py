import flask
import pickle
import pandas as pd
import numpy as np

# Use pickle to load in the pre-trained model
with open(f'modelb.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')


# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Extract the input
        jobcat = flask.request.form['jobcat']
        degree = flask.request.form['degree']
        major = flask.request.form['major']
        industry = flask.request.form['industry']
        workexperience = flask.request.form['workexperience']
        dis = flask.request.form['dis']

        Input=[jobcat,degree,major,industry,workexperience,dis]
        if Input[0] == 'CEO':
            Encoded0 = np.array([[1., 0., 0., 0., 0., 0., 0., 0.]])
        if Input[0] == 'CFO':
            Encoded0 = np.array([[0., 1., 0., 0., 0., 0., 0., 0.]])
        if Input[0] == 'CTO':
            Encoded0 = np.array([[0., 0., 1., 0., 0., 0., 0., 0.]])
        if Input[0] == 'JANITOR':
            Encoded0 = np.array([[0., 0., 0., 1., 0., 0., 0., 0.]])
        if Input[0] == 'JUNIOR':
            Encoded0 = np.array([[0., 0., 0., 0., 1., 0., 0., 0.]])
        if Input[0] == 'MANAGER':
            Encoded0 = np.array([[0., 0., 0., 0., 0., 1., 0., 0.]])
        if Input[0] == 'SENIOR':
            Encoded0 = np.array([[0., 0., 0., 0., 0., 0., 1., 0.]])
        if Input[0] == 'VICE_PRESIDENT':
            Encoded0 = np.array([[0., 0., 0., 0., 0., 0., 0., 1.]])

        if Input[1] == 'BACHELORS':
            Encoded1 = np.array([[1., 0., 0., 0., 0.]])
        if Input[1] == 'DOCTORAL':
            Encoded1 = np.array([[0., 1., 0., 0., 0.]])
        if Input[1] == 'HIGH_SCHOOL':
            Encoded1 = np.array([[0., 0., 1., 0., 0.]])
        if Input[1] == 'MASTERS':
            Encoded1 = np.array([[0., 0., 0., 1., 0.]])
        if Input[1] == 'NONE':
            Encoded1 = np.array([[0., 0., 0., 0., 1.]])

        if Input[2] == 'BIOLOGY':
            Encoded2 = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]])
        if Input[2] == 'BUSINESS':
            Encoded2 = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        if Input[2] == 'CHEMISTRY':
            Encoded2 = np.array([[0., 0., 1., 0., 0., 0., 0., 0., 0.]])
        if Input[2] == 'COMPSCI':
            Encoded2 = np.array([[0., 0., 0., 1., 0., 0., 0., 0., 0.]])
        if Input[2] == 'ENGINEERING':
            Encoded2 = np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0.]])
        if Input[2] == 'LITERATURE':
            Encoded2 = np.array([[0., 0., 0., 0., 0., 1., 0., 0., 0.]])
        if Input[2] == 'MATH':
            Encoded2 = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0.]])
        if Input[2] == 'NONE':
            Encoded2 = np.array([[0., 0., 0., 0., 0., 0., 0., 1., 0.]])
        if Input[2] == 'PHYSICS':
            Encoded2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        if Input[3] == 'AUTO':
            Encoded3 = np.array([[1., 0., 0., 0., 0., 0., 0.]])
        if Input[3] == 'EDUCATION':
            Encoded3 = np.array([[0., 1., 0., 0., 0., 0., 0.]])
        if Input[3] == 'FINANCE':
            Encoded3 = np.array([[0., 0., 1., 0., 0., 0., 0.]])
        if Input[3] == 'HEALTH':
            Encoded3 = np.array([[0., 0., 0., 1., 0., 0., 0.]])
        if Input[3] == 'OIL':
            Encoded3 = np.array([[0., 0., 0., 0., 1., 0., 0.]])
        if Input[3] == 'SERVICE':
            Encoded3 = np.array([[0., 0., 0., 0., 0., 1., 0.]])
        if Input[3] == 'WEB':
            Encoded3 = np.array([[0., 0., 0., 0., 0., 0., 1.]])

        EncodedInput = np.append(Encoded0, Encoded1, axis=1)
        EncodedInput = np.append(EncodedInput, Encoded2, axis=1)
        EncodedInput = np.append(EncodedInput, Encoded3, axis=1)
        EncodedInput = np.append(EncodedInput, np.array([[Input[4]]]), axis=1)
        EncodedInput = np.append(EncodedInput, np.array([[Input[5]]]), axis=1)

        # Make DataFrame for model
        input_variables = pd.DataFrame([[jobcat, degree, major, industry, workexperience, dis]],
                                       columns=['jobcat', 'degree', 'major','industry', 'workexperience', 'dis'],
                                                                             index=['input'])



        # Get the model's prediction
        prediction = model.predict(EncodedInput)

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'jobcat': jobcat,
                                                     'degree': degree,
                                                     'major': major,
                                                     'industry':industry,
                                                     'workexperience':workexperience,
                                                     'dis':dis},
                                     result=str(int(prediction[0][0]))+'k',
                                     )


if __name__ == '__main__':
    app.run()