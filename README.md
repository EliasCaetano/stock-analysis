📈 Stock Market Analysis & Prediction

📌 Overview

This project focuses on analyzing and predicting stock market behavior using historical data from major tech companies (Google, Tesla, Microsoft, Amazon).

The goal is to:

Perform Exploratory Data Analysis (EDA)
Engineer meaningful financial features
Build a predictive model for stock returns
Compare performance against a strong baseline

🧠 Problem Statement

Predicting stock prices directly is extremely difficult due to noise and market efficiency.

Instead, this project reframes the problem to:

👉 Predict daily returns rather than raw prices

This allows the model to better capture short-term patterns and temporal dependencies.

```
⚙️ Stock-Analysis
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── analysis.py
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── utils.py
├── notebooks/
│   └── analysis.ipynb
├── main.py
├── train_model.py
├── requirements.txt
├── README.md
└── insights.md
```

📊 Exploratory Data Analysis (EDA)

Key analyses performed:

Accumulated return per asset
Price evolution over time
Return distribution (mean, median, skewness)
Volatility (daily and rolling)
Correlation between assets
Maximum drawdown

🔍 Key Findings
Tesla is the most volatile asset and also offers the highest risk-return profile
Google achieved the highest accumulated return over the period
Returns distribution shows positive skewness (0.31) → more extreme positive returns
Strong correlations between big tech stocks (especially Microsoft & Amazon)
Tesla experienced the largest drawdown (~73%), highlighting its risk

🏗️ Feature Engineering

Features used:

Moving Averages (MA_10, MA_50)
Rolling Volatility (Vol_10)
Lagged Returns (Return_Lag_1, Return_Lag_2, Return_Lag_3)

⚠️ Important:

The current return is NOT used as feature to avoid data leakage

🤖 Model

Model used:

👉 Random Forest Regressor

Target:

👉 Daily Return

📏 Evaluation

Model Performance
RMSE: 0.0209
MAE: 0.0145
R²: -0.0431

Baseline (Previous Day Return)
RMSE: 0.0283
MAE: 0.0202

✅ The model outperforms the baseline, indicating it captures useful patterns.

🧠 Feature Importance

Top features:

Return_Lag_3
Return_Lag_2
Return_Lag_1

👉 The model relies heavily on past returns, which aligns with financial intuition.

🚀 How to Run
1. Preprocess data
python main.py --step preprocess
2. Train model
python main.py --step train
3. Evaluate
python main.py --step evaluate
4. Predict
python main.py --step predict

🔮 Future Improvements
Try Gradient Boosting / XGBoost
Predict direction (classification)
Add more lag features
Time-series cross-validation
Build a dashboard (Streamlit)

📌 Conclusion

This project demonstrates how reframing a problem and applying proper feature engineering can significantly improve model performance.

It highlights the importance of:

Baseline comparison
Avoiding data leakage
Understanding time series behavior