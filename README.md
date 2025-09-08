

# 📈 Stock Market Prediction App

A **Stock Price Prediction Web Application** built with **Streamlit**, **Keras (TensorFlow)**, and **Yahoo Finance API**.  
This app allows users to input a stock ticker (e.g., `GOOG`, `AAPL`) and visualize stock data along with machine learning–based predictions.

---

## 🚀 Features
- Fetches historical stock data using [yfinance](https://pypi.org/project/yfinance/).
- Preprocesses and scales data using `MinMaxScaler`.
- Uses a **pre-trained deep learning model** (`.keras`) to predict stock closing prices.
- Displays:
  - Historical stock data table
  - Moving Average trend (50-day)
  - Predicted vs Actual stock prices (future extension)

---

## 🛠️ Tech Stack
- **Python 3.9+**
- [Streamlit](https://streamlit.io/) – frontend web framework
- [Keras / TensorFlow](https://keras.io/) – deep learning model
- [yfinance](https://pypi.org/project/yfinance/) – stock data fetching
- [scikit-learn](https://scikit-learn.org/) – scaling data
- [matplotlib](https://matplotlib.org/) – data visualization
- [numpy & pandas](https://pandas.pydata.org/) – data processing

---

## 📂 Project Structure


```
predict-stock-price/
│
├── frontend/
│   └── app.py                # Streamlit application
│
├── stock-prediction-model.keras   # Pre-trained model (LSTM/GRU/etc.)
│
├── requirements.txt          # Dependencies list
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mrutyu-mark-v/predict-stock-price.git
cd predict-stock-price/frontend
````

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
pandas
yfinance
scikit-learn
matplotlib
streamlit
tensorflow
keras
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📊 Usage

1. Enter a stock symbol (e.g., `GOOG`, `AAPL`, `TSLA`) in the input box.
2. The app will:

   * Fetch stock history from Yahoo Finance.
   * Show raw stock data.
   * Plot 50-day moving average vs actual price.
   * Predict stock price trends using the trained model.
3. Adjust the stock symbol to analyze different companies.

---


## 📌 Future Improvements

* Add 100-day and 200-day moving averages.
* Plot **Actual vs Predicted Prices** side by side.
* Integrate live stock prediction with scheduled retraining.
* Deploy app on **Streamlit Cloud / AWS / Heroku**.

---

## 🧑‍💻 Author

* MRUTYUNJAYA
* [LinkedIn](https://linkedin.com/in/mrutyunjaya-bagha-63254b30b) | [GitHub](https://github.com/Mrutyu-mark-v)
