# Stock Price Prediction using LSTM

A deep learning project that predicts stock prices using Long Short-Term Memory (LSTM) neural networks. This model analyzes historical stock data and predicts future price movements using TensorFlow/Keras.

## 📊 Project Overview

This project implements an LSTM-based model to forecast stock prices with high accuracy. The model is trained on historical stock price data and uses a 100-day lookback window to predict the next day's price.

**Stock Used**: TCS (Tata Consultancy Services) - `TCS.NS`  
**Data Period**: 2010-01-01 to Present

## 🎯 Key Features

- **Data Fetching**: Automatically downloads historical stock data using Yahoo Finance
- **Data Preprocessing**: Normalizes data using MinMaxScaler
- **Moving Averages**: Calculates and visualizes 100-day and 200-day moving averages
- **LSTM Model**: Deep neural network with 4 LSTM layers
- **Model Evaluation**: Comprehensive metrics including MAE, RMSE, and R² score
- **Visualization**: Plots for actual vs predicted prices and cumulative returns

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep Learning Framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Matplotlib** - Data visualization
- **yfinance** - Yahoo Finance data fetching
- **Jupyter Notebook** - Interactive analysis environment

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Stock-Price-Prediction-Basic.git
   cd Stock-Price-Prediction-Basic
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas numpy matplotlib yfinance tensorflow scikit-learn
   ``

## 📈 Model Architecture

The LSTM model consists of:

| Layer | Units | Activation | Dropout |
|-------|-------|-----------|---------|
| LSTM Layer 1 | 50 | ReLU | 0.2 |
| LSTM Layer 2 | 60 | ReLU | 0.3 |
| LSTM Layer 3 | 80 | ReLU | 0.4 |
| LSTM Layer 4 | 120 | ReLU | 0.5 |
| Output (Dense) | 1 | Linear | - |

**Optimizer**: Adam  
**Loss Function**: Mean Squared Error (MSE)  
**Lookback Window**: 100 days

## 📊 Data Processing Pipeline

1. **Data Loading**: Downloads historical stock data from Yahoo Finance (2010-present)
2. **Data Cleaning**: Removes null values and drops unused columns
3. **Train-Test Split**: 70% training, 30% testing
4. **Normalization**: MinMaxScaler (0-1 range) for better model convergence
5. **Sequence Creation**: Creates 100-day sequences for LSTM input
6. **Prediction**: Generates predictions on test set and inverse transforms to original scale

## 📉 Model Evaluation Metrics

The model is evaluated using:

- **Mean Absolute Error (MAE)**: Measures average absolute difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **Cumulative Returns Comparison**: Visualizes the relationship between actual and predicted cumulative returns

## 🔮 Predictions

The model generates:
- Point predictions for each test date
- Visualization of predicted vs actual stock prices
- Analysis of prediction accuracy

## 🎓 How It Works

1. **Historical Data**: Fetches long-term historical stock price data
2. **Feature Engineering**: Uses closing prices with 100-day moving averages
3. **Training**: Model learns patterns from 70% of the data
4. **Validation**: Tests predictions on remaining 30% of data
5. **Prediction**: Uses 100 previous days to predict the next day's price

## ⚠️ Limitations & Considerations

- **Past Performance**: Historical data doesn't guarantee future results
- **Market Variables**: Model doesn't account for external factors (news, economic events)
- **Stock-Specific**: Currently trained on TCS.NS data (can be modified for other stocks)
- **Accuracy**: Model performance may vary with different time periods and stocks
- **Real-World Risk**: Should not be used as sole basis for investment decisions

## 🔧 Customization & Using Different Stock Symbols

### Changing Stock Ticker

The model can be easily adapted to predict prices for any stock available on Yahoo Finance. Simply modify the ticker symbol in the cell that loads data:

```python
data = load_data("TCS.NS")  # Change this to any stock ticker
```

### Stock Ticker Examples

#### US Stocks (US Exchange)
```python
data = load_data("AAPL")    # Apple Inc.
data = load_data("GOOGL")   # Alphabet (Google)
data = load_data("MSFT")    # Microsoft
data = load_data("AMZN")    # Amazon
data = load_data("META")    # Meta (Facebook)
```

#### Indian Stocks (NSE - National Stock Exchange)
```python
data = load_data("TCS.NS")      # Tata Consultancy Services (default)
data = load_data("INFY.NS")     # Infosys
data = load_data("RELIANCE.NS") # Reliance Industries
data = load_data("ITC.NS")      # ITC Limited
data = load_data("WIPRO.NS")    # Wipro
```

#### Other International Stocks
```python
data = load_data("ASIANPAINT.NS")  # Asian Paints (India)
data = load_data("0001.HK")        # Hang Seng Bank (Hong Kong)
data = load_data("SAP.DE")         # SAP (Germany)
data = load_data("BARC.L")         # Barclays (UK)
```

### Step-by-Step: Predicting a Different Stock

1. **Locate the data loading cell** (around the 3rd code cell):
   ```python
   data = load_data('TCS.NS')  # Original line
   ```

2. **Replace with your desired ticker**:
   ```python
   data = load_data('INFY.NS')  # For Infosys
   # or
   data = load_data('AAPL')     # For Apple
   ```

3. **Run all cells sequentially** to:
   - Fetch new stock data
   - Preprocess and normalize
   - Train the LSTM model
   - Generate predictions for the new stock

### Adjusting Hyperparameters

For different stocks, you may want to optimize:

- **LSTM Units**: Increase for more complex patterns
  ```python
  model.add(LSTM(units = 100, ...))  # Increased from 50
  ```

- **Dropout Rates**: Prevent overfitting
  ```python
  model.add(Dropout(0.3))  # Increase if overfitting occurs
  ```

- **Training Epochs**: Number of training iterations
  ```python
  model.fit(x_train, y_train, epochs = 150)  # Increase from 100
  ```

- **Train-Test Split**: Ratio for training vs testing
  ```python
  train = pd.DataFrame(data[0:int(len(data)*0.75)])  # 75% train, 25% test
  test = pd.DataFrame(data[int(len(data)*0.75): int(len(data))])
  ```

### Data Period Customization

The **dataset size directly impacts model performance**. You can control this by adjusting the `START` date in the data loading cell:

```python
START = "2010-01-01"  # Default: 10+ years of data
TODAY = date.today().strftime("%Y-%m-%d")
```

#### Date Range Examples & Expected Impact

#### How Dataset Size Affects the Model

- **Small datasets (< 500 points)**: 
  - Risks overfitting (memorizes instead of learning patterns)
  - Less reliable predictions
  - Faster training
  
- **Medium datasets (500-2000 points)**:
  - Good balance between speed and accuracy
  - Learns meaningful patterns
  - Recommended for most use cases
  
- **Large datasets (2000+ points)**:
  - Better generalization
  - Captures long-term market cycles
  - Takes longer to train
  - More robust predictions

#### Recommended Dataset Sizes for Different Scenarios

```python
# For quick experimentation / testing the model
START = "2023-01-01"  # Recent 3 years of data

# For balanced performance (recommended)
START = "2018-01-01"  # 8 years of data

# For comprehensive long-term predictions
START = "2005-01-01"  # Full Timeframe

# For short-term trading predictions
START = "2022-01-01"  # 4 years of recent market data
```

**General Rule**: Use at least 4-5 years of data for reliable predictions. More data = better patterns learned, but slower training.
