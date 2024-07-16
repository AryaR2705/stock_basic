from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Function to predict stock prices for a given date
def predict_stock_prices(input_date):
    # Convert the input date to ordinal
    date_ordinal = datetime.strptime(input_date, '%Y-%m-%d').toordinal()
    date_ordinal = np.array([[date_ordinal]])

    # Dictionary to store predictions for each stock
    predictions = {}

    for stock in ['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']:
        # Load the model
        model = joblib.load(f'{stock}_model.pkl')
        
        # Predict the stock price
        predicted_price = model.predict(date_ordinal)
        predictions[stock] = predicted_price[0]

    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_date = request.form['date']
    predictions = predict_stock_prices(input_date)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
