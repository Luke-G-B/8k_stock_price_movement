# Buy the Rumor, Sell the News

*This project is intended to explore machine learning concepts using stock market data and not to provide trading recommendations.*

## Predicting stock price directional movement after 8-K filings

A form 8-K is a general form used by publicly traded companies in the United States to notify shareholders of specific events that may be important. These events can include bankruptcy, the issuing of new shares, change in accountants, and more. [Read more about 8-K's at wikipedia](https://en.wikipedia.org/wiki/Form_8-K).

This project looks to explore the information gain from 8-K text through natural language processing. Using this, along with technical indicators about the stock price's recent performance, a machine learning model is used, aiming to predict directional movement following the filing.

## The data

* More than 48,000 8-K's scraped from sec.gov for all S&P 500 companies
* Stock price data collected from the Alpha Vantage API
  * Relative Strength index
  * Consumer Channel Index
  * On Balance Volume
  * Moving Standard Deviation
  * Moving Rate of Change

## The models

8-K text --> term frequency vectors --> Multinomial Naive Bayes classifier predicting up or down movement

The Naive Bayes classification is then included as a feature, along with the above technical indicators to prediction directional movement

## Evaluation

## Conclusion
