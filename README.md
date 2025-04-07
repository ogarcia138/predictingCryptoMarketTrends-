# Binary Classification Predicting Crypto Market Trends using Machine Learning Models
This repository contains the project for the CSC481 class. The project Binary Classification Predicting Crypto Market Trends using Machine Learning Models

**Abstract**  
The cryptocurrency market is highly volatile, making it challenging for traders and investors to make informed decisions. This project aims to leverage machine learning models to perform binary classification on social media content, predicting whether cryptocurrency prices will rise or fall within a specified timeframe based on this data. By analyzing historical market trends and integrating relevant financial indicators, our model seeks to enhance predictive accuracy and provide data-driven insights for traders. In addition to these traditional signals, we employ FinBERT—a pre-trained transformer model specialized for financial text—to quantify social media sentiment, capturing the emotional factors that can influence rapid market fluctuations. We will implement our solution using Python and utilize X, Kraken’s, and Reddit’s APIs to pull real-time market data, ensuring our forecasts remain up-to-date and dynamic.

# Introduction
**Purpose**  
This project leverages machine learning models to perform binary classification on social media content, predicting whether cryptocurrency prices will rise or fall within a specified timeframe. By analyzing historical market trends and integrating relevant financial indicators, the model aims to enhance predictive accuracy and offer data-driven insights for traders. The solution utilizes Python, along with real-time data from X, Kraken’s, and Reddit’s APIs, ensuring dynamic and up-to-date predictions.

**Problem Statement**  
Cryptocurrency markets are highly volatile, making it challenging for traders and investors to make informed decisions. Fluctuations can occur rapidly due to market sentiment and external factors, often leading to significant financial risk. By combining machine learning techniques with social media sentiment analysis, this project seeks to address these challenges and support more confident, evidence-based trading strategies.

# Literature Review
Background on Cryptocurrency Forecasting

Research on cryptocurrency price prediction has gained momentum in recent years, primarily due to the market’s rapid growth and volatility. Traditional methods often rely on time-series analysis (e.g., ARIMA, GARCH) or technical indicators (e.g., MACD, RSI) for trend forecasting. While these methods can capture historical patterns, they sometimes struggle to incorporate the real-time market sentiment that frequently drives short-term price fluctuations.

More recent studies leverage machine learning and deep learning techniques—such as Random Forest, XGBoost, and Recurrent Neural Networks (RNNs)—to improve predictive accuracy. These approaches excel at handling larger feature spaces and can adapt to complex, non-linear relationships within the data. However, a common gap in the literature is the limited integration of sentiment data, which often holds key insights into rapid market changes driven by collective investor perception.

By combining both technical and sentiment-based features, our approach aims to address this gap, offering a more holistic view of the market. This could potentially reduce the risk of missing sudden trend shifts caused by social media events or announcements that purely technical models might overlook.

**Sentiment Analysis in Finance**

The rationale for using social media sentiment data lies in the growing recognition that market movements are not solely driven by technical factors but also by investor psychology. Platforms like Reddit, X (formerly Twitter), and other social channels can significantly influence short-term price actions, as large communities of traders often synchronize their decisions based on trending discussions or influential opinions.

Several successful studies have illustrated the usefulness of sentiment-driven models for predicting stock price movements, with improvements in short-term accuracy when compared to models relying solely on historical prices. However, these results also highlight limitations:

-Noise and Bot Activity: Social media data often contain spam or automated posts that can skew sentiment analysis.

-Contextual Understanding: Simply labeling a post as positive or negative may not suffice if the context or sarcasm is missed by the sentiment engine.

-Rapidly Evolving Language: Internet slang and meme-driven cultures (common in crypto communities) can require specialized NLP models.

Despite these challenges, the potential benefits of integrating sentiment signals remain significant. As the cryptocurrency market is particularly susceptible to news cycles and collective online behaviors, robust sentiment analysis can capture market hype or fear before it’s fully reflected in price charts.

# Data Acquisition

**Data Sources**
This project integrates data from multiple sources to provide both historical and real-time context for cryptocurrency market movements:
1. Cryptocurrency Exchanges
   
      -Kraken: Primary sources for OHLC (Open, High, Low, Close) price data and trading volume.
      -Potentially other exchanges (e.g., Coinbase, Binance) for broader coverage or alternative trading    pairs
2. Social Media Platforms
   
      -Reddit: Data extracted from cryptocurrency-focused subreddits (e.g., r/CryptoCurrency) via the       Reddit API or PRAW.

      -X (formerly Twitter): Tweets mentioning specific cryptocurrencies or hashtags fetched via Twitter/X's API or libraries like Tweepy.

   
3. Sentiment Analysis Tool

   
      -FinBERT: Used to perform sentiment scoring on the collected text data. Vader is particularly effective for short, social-media style text, providing scores for positive, negative, neutral, and an overall compound sentiment metric.

By merging exchange price data with social media sentiment data, this project aims to capture both objective market signals and subjective investor mood, enabling more nuanced cryptocurrency price predictions.

**Data Collection Process**

1. Frequency

   
     -Real-Time Retrieval: For continuous updates (e.g., every minute or every 5 minutes), Python scripts pull the latest market data from Kraken and new posts or tweets from Reddit and X, respectively.  

   -Historical Data: Collected on a daily/hourly basis to build a robust dataset for backtesting, training, and validation.

   
3. Methods & Tooling

   
   API Endpoints & Libraries:  
  
      -Kraken's API for near real-time and historical crypto price data. 
   
      -Reddit's API/PRAW for subreddit posts, comments, and discussion threads.  

      -Twitter/X's API /Tweepy for streaming or batch retrieval of tweets.  
       
    FinBERT for Sentiment:  
       
      -Used to perform sentiment scoring on the collected text data. FinBERT is a pre-trained transformer model specifically fine-tuned for financial text, providing robust sentiment insights (positive, negative, neutral) that can be particularly valuable for understanding market-related social media content.  
      
      -The resulting sentiment scores (positive, negative, neutral, and compound) are stored alongside the timestamp and relevant cryptocurrency references.  

       
   Scripts & Scheduling:  

  
      -Custom Python scripts handle data ingestion, sentiment processing, and caching.  
  
      -Cron jobs (or similar schedulers) run at fixed intervals to automate the data-fetching routine and ensure up-to-date information.  

By systematically combining market data (OHLC) and sentiment scores (from VADER), the project maintains a dynamic and comprehensive view of the cryptocurrency landscape. This foundation enables both effective model training and real-time prediction for traders and investors.

# Data Preprocessing


**Data Cleaning**

1. Missing or Duplicated Entries


   -After retrieving both market (OHLC) and social media data, we check for missing values (e.g., null timestamps, missing prices) or duplicated records.

   -Duplicates are dropped to avoid skewed counts in either the price or sentiment dataset. Missing values -if minimal- may be imputed (e.g., forward-fill for time series) or removed entirely if they exceed a threshold.

2. Corrupted Rows & Timestamp Alignment

   -Inconsistent or corrupted rows (e.g., malformed text posts, invalid price points) are either corrected or filtered out.

   -We align market data and sentiment data based on timestamps, ensuring each sentiment score is associated with the nearest relevant trading interval. This may involve rounding or interpolating timestamps to the closest minute, hour, or day, depending on the analysis window.

**Normalization & Transformation**

1. Scaling Numerical Features

      -Features like trading volume, price changes, or technical indicators can vary significantly in magnitude. We apply common scaling techniques such as StandardScaler (mean = 0, variance = 1) or MinMaxScaler (scaled to [0,1]) to normalize these features.

      -Scaling helps certain machine learning algorithms converge faster and avoids bias towards high-magnitude variables.

2. Text Cleaning

      -For social media posts, we remove URLs, punctutation, and other non-text characters.

      -We convert all text to lowercase and optionally remove stop words (common words like "the", "and", etc.) to reduce noise.

      -Depending on the model, more advanced cleaning (e.g., lemmatization or stemming) may be applied to group similar word forms.


**Sentiment Extraction**

1. From Raw Text to Sentiment Scores

      -Each collected post or tweet is passed through a sentiment analysis tool, producing metrics such as positive, negative, neutral, or a compound sentiment score.

      -We store these sentiment scores alongside the timestamp and any metadata (e.g., user, subreddit, or hashtags).

2. Model & Libraries

      -FinBERT is our primary tool for analyzing financial or market-related text, including short-form social media content. FinBERT, a transformer-based model specifically fine-tuned for finance, provides more context-aware sentiment insights (positive, negative, neutral) compared to general-purpose sentiment analysis tools.


      -We may optionally compare results from TextBlob or transformer-based models (e.g., BERT) to see if advanced NLP methods improve sentiment accuracy.


By applying these cleaning, normalization, and sentiment-extraction steps, we create a consistent, high-quality dataset. This ensures our machine learning models can effectively learn patterns and generate reliable predictions for cryptocurrency price movements.

# Exploratory Data Analysis (EDA)

**Initial Insights**

1. Price Trend Visualization

      -We generate line plots of the cryptocurrency's closing prices over time (using libraries like Matplotlib or Plotly)

      -These visualizations help reveal broad trands, volatility patterns, and potential support/resistance levels.

2. Correlation with Technical Indicators

      -Using correlation matrices or scatter plots, we explore how MACD, RSI, or other technical indicators correlate with price movements.

      -This step guides us in identifying which indicators might be most predictive of future price changes.

3. Sentiment vs. Price

      -We overlay or plot sentiment scores (e.g., from VADER) alongside the price charts to see if spikes in positive/negative sentiment align with major price moves.

      -This can offer early insights into how closely sentiment data tracks or foreshadows volatility.

**Statistical Summaries**

1. Descriptive Statistics

      -For price data (OHLC, returns, trading volume) and sentiment variables (positive, negative, neutral scores), we compute standard summary metrics:

      -Mean, Median, Variance, Standard Deviation

      -Min/Max values and Interquartile ranges

      -These metrics provide a baseline understanding of the data's distribution and spread.

2. Outlier Detection & Patterns

      -We may look at box plots or histograms to identify potential outliers in both price and sentiment data.

      -Recognizing outliers (e.g., extreme sentiment spikes or unusually high volume) can guide data cleaning or model weighting.

By combining visual analyses with basic statistical checks, we gain a deeper understanding of our dataset's structure. These EDA findings often inform feature engineering choices and model selection, ensuring we capture the most relevant predictors of cryptocurrency price movements.

# Feature Engineering

**Technical Indicators**

1. Indicator List

      -MACD (Moving Average Convergence Divergence): Captures momentum by comparing short-term and long-term exponential moving averages (EMAs).

      -RSI (Relative Strength Index): Measures the speed and magnitude of price movements to identify overbought or oversold conditions.

      -Moving Averages (SMA/EMA): Smooth out short-term fluctuations in price data, providing clearer long-term trends.

      -Others: We may include Bollinger Bands, Stochastic Oscillator, or volume-based indicators if they offer additional predictive power.

2. Window Size & Hyperparameters

      -We experiment with various lookback periods (e.g., 14-day RSI or 12/26-day MACD) based on common technical analysis standards.

      -These parameters can also be tuned during the model training process to find the optimal timeframe for each indicator.

**Sentiment Features**

1. Aggregating Sentiment

      -We take individual FinBERT scores (positive, negative, neutral, compound) and compute daily or hourly averages to reflect overall market mood.

      -In some cases, weighted average (e.g., weighting sentiment by social media engagement or trading volume) may better capture the influence of major posts or tweets.

2. Combining Multiple Sources

      -If using both Reddit and X (formerly Twitter), we may merge or compare sentiment data to see if one platform provides stronger signals.

      -We can derive composite sentiment scores by blending multiple signals to create a single predictive feature.

**Dimensionality Reduction**


   -If dealing with high-dimensional sentiment data (e.g., word embeddings, multiple platforms), methods like PCA (Principal Component Analysis) or t-SNE can be applied to condense the feature space.

   -Reducing complexity may mitigate overfitting and can speed up training without substantially sacrificing predictive accuracy.

By engineering features from both technical indicators and sentiment data, we aim to capture complementary perspectives on market movements. These engineered signals play a crucial rolde in enhancing the performance of our predictive models.

# Model Selection & Architecture

**Choice of Algorithms**

1. Logistic Regression

      -A straighforward baseline for binary classification with interpretability and relatively low computational cost.

      -Works well on linearly seperable data and can highlight how different features contribute to the prediction (via coefficients).

2. Random Forest

      -Ensemble of decision trees that can handle non-linear relationships and is relatively robut to outliers.

      -Often achieves strong performance with minimal hyperparameter tuning; provides feature importance insights.

3. XGBoost

      -A gradient-boosting framework known for high efficiency and strong performance on tabular data.

      -Handles both linear and non-linear relationships, regularizes more effectively than simpler ensemble methods, and scales well to large datasets.

4. Neural Networks

      -Capable of learning complex, non-linear patterns, especially useful if you have high-dimensional data (e.g., embeddings, multiple sentiment scores).

      -Flexibility in architecture (e.g., fully connected networks, LSTM for time series) can capture intricate market or sentiment relationships.

**Implementation Details**

Libraries & Frameworks
      
   -scikit-learn: Used for Logistic Regression, Random Forest, and preliminary data processing (train-test splits, scaling).
   
   -XGBoost: Standalone library for a gradient boosting and advanced hyperparameter tuning.

   -TensorFlow or PyTorch: Implement and train the Neural Network architecture.

Hyperparameters & Configurations

   -Logistic Regression:
       
   --Typically uses solver='lbfgs' or solver ='saga' for large datasets, with L2 regularization.
   
      
   -Random Forest:
   
   
   --Key hyperparameters include n_estimators (number of trees) and max_depth
   
   
   -XGBoost:
   
   
   --Commonly tuned parameters are learning_rate, n_estimators, max_depth, and colsample_bytree
   
   
   -Neural Network:
       
       
   --May involve architectures with multiple dense layers, dropout for regularization, batch size, and learning rate.
   
   
   --Optimizers like Adam or SGD are frequently used, along with ReLU activation functions.

By employing a diverse set of models-ranging from interpretable (Logistic Regression) to highly flexible (Neural Networks)-the project aims to identify the optimal balance between predictive accuracy, training speed, and interpratibility.

# Model Training & Hyperparameter Tuning

**Training Procedure**

1. Data Splitting

      -The dataset is divided into training, validation, and test subjects. Typically, an 80/10/10 or 70/15/15 split is used for balancing training and evaluation.

      -In certain cases, k-fold cross-validation may replace or supplement a fixed split to better estimate out-of-sample performance.

2. Time Series Considerations

      -For price data, we employ walk-forward validation or a rolling window approach to respect the chronological order. This prevents data leakage by ensuring that future data doesn't influence past predictions.

      -The validation set is updated incrementally, aligning with real-world trading scenarios where new data continously arrives.

**Hyperparameter Tuning**

1. Tuning Techniques

      -Grid Search: Explores a predefined parameter (e.g., max_depth in Random Forest, learning_rate in XGBoost).

      -Random Search: Randomly samples a perameter space for faster discovery of good regions.

      -Bayesian Optimization: Uses probabilistic methods to guide the search for optimal hyperparameters, often more efficient than brute-force grid search.

2. Tracking Improvements

      -Performance Metrics: Monitored primarily through metrics like accuracy, precision, recall, F1-score, or ROC-AUC(relevant)

      -Logging & Visualization: Tools such as TensorBoard (for neural networks) or custom logging scripts record how changes in hyperparameters affect model performance over epochs.

      -Iterative Refinement: Promising hyperparameter sets are validated on the seperate test dataset or via cross-validation to confirm that performance gains aren't due to overfitting.

By systematically refining both model architecture and hyperparameters, we aim to balance predictive accuracy, training speed, and generalization. This iterative process helps ensure the final model is both robust and well-tuned for real-world cryptocurrency market predictions.


# Model Evaluation #

**Evaluation Metrics**
   1. Accuracy

      -Represents the proportion of correct predictions (both price up or down) out of the total number of predictions. Useful for a quick overview but can be misleading if the dataset is imbalanced.

   2. Precision & Recall

      -Precision (Positive Predictive Value): Out of all predicted price increases, how many were correct?

      -Recall (True Positive Rate): Out of all actual price increases, how many did we predict correctly?

      -These metrics are especially valuable if misclassifications have different financial implications (e.g., a missed rise vs. a false alarm).

   3. F1-Score

      -The harmonic mean of Precision and Recall, Providing a balanced measure when class distribution is uneven or when both false positives and false negatives are costly.

**Results Comparison**


Model'               'Accuracy'               'Precision'               'Recall'               'F1-Score


Logistic Regression	  


Random Forest	         


XGBoost	              


Neural Network	        

1. Performance Insights

      -Summarize which model achieved the highest accuracy or balanced metrics

2. Trade-Offs

     -Speed vs. Accuracy: Some models(e.g., Random Forest) may train quickly but offer slightly lower accuracy, while Neural Networks might be more computationally expensive yet capture deeper patterns.

      -Overfitting Concerns: If a model exhibits strong performance on the training set but weaker performance on the test set, it may be overfitting. Techniques like cross-validation or regularization can mitigate this.

   By systematically comparing these metrics, we ensure a transparent and data-driven approach to selecting the final model for cryptocurrency price prediction.
