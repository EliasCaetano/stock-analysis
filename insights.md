🧠 Storytelling & Insights

1. Initial Hypothesis

The initial approach was to predict stock prices directly. However, results were poor and underperformed a naive baseline.

This indicated that:

👉 The model was not capturing meaningful patterns.

2. Problem Reframing

The strategy shifted to predicting returns instead of prices.

Why?

Returns are more stationary
Easier for models to learn
More aligned with financial theory

3. Feature Engineering Breakthrough

The key improvement came from adding lagged returns.

This allowed the model to:

Capture short-term temporal dependencies
Learn patterns from recent market behavior

4. Results Evolution

Before:

Model worse than baseline ❌

After:

Model beats baseline ✅

This confirms that:

👉 Feature engineering had a direct impact on performance

5. Market Insights

Tesla shows extreme volatility → high risk/high reward
Big tech stocks are strongly correlated
Positive skewness suggests upside potential events

6. Key Learning

The biggest improvement did not come from changing the model, but from changing the problem formulation and features.

7. Final Takeaway

This project demonstrates a realistic data science workflow:

Start simple
Fail against baseline
Investigate
Improve features
Re-evaluate