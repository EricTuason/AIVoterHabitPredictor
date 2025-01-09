from flask import Flask, render_template, request
import nonvoters as nv

app = Flask('nonvoters')

@app.route('/')
def show_predict_form():
    #write your function that loads the model
    nonvoters = nv.Nonvoters()
    model_params, model = nonvoters.find_model() #you can use pickle to load the trained model
    predicted = nonvoters.find_predictions(model)
    image_buffer = nonvoters.save_model()
    accuracy, variance, mse, precision, recall = nonvoters.find_stats()
    return render_template('predictorform.html', predicted=predicted, buffer_encode=image_buffer, best_model=model_params, accuracy=accuracy, variance=variance, mse=mse, precision=precision, recall=recall)

@app.route('/results', methods=['POST'])
def show_result_form():
    if request.method == 'POST':
        education = request.form['education']
        race = request.form['race']
        gender = request.form['gender']
        income = request.form['income']
        return render_template('resultsform.html', model_output=education+race+gender+income)

app.run("localhost", "9999", debug=True) 