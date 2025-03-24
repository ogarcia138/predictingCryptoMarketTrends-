# Binary Classification Predicting Crypto Market Trends using Machine Learning Models
This repository contains the project for the CSC481 class. The project Binary Classification Predicting Crypto Market Trends using Machine Learning Models

Abstract
The cryptocurrency market is highly volatile, making it challenging for traders and investors to make informed decisions. This project aims to leverage machine learning models to perform binary classification on social media content, predicting whether cryptocurrency prices will rise or fall within a specified timeframe based off this data. By analyzing historical market trends and integrating relevant financial indicators, our model seeks to enhance predictive accuracy and provide data-driven insights for traders. We will implement our solution using Python, utilize X, Kraken’s, and Reddit's API to pull real-time market data, ensuring up-to-date and dynamic predictions
# Introduction

**Purpose**  
This project leverages machine learning models to perform binary classification on social media content, predicting whether cryptocurrency prices will rise or fall within a specified timeframe. By analyzing historical market trends and integrating relevant financial indicators, the model aims to enhance predictive accuracy and offer data-driven insights for traders. The solution utilizes Python, along with real-time data from X, Kraken’s, and Reddit’s APIs, ensuring dynamic and up-to-date predictions.

**Problem Statement**  
Cryptocurrency markets are highly volatile, making it challenging for traders and investors to make informed decisions. Fluctuations can occur rapidly due to market sentiment and external factors, often leading to significant financial risk. By combining machine learning techniques with social media sentiment analysis, this project seeks to address these challenges and support more confident, evidence-based trading strategies.

#Literature Review
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
