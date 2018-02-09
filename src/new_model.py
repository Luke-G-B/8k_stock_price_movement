import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score


class Prepare8kCorpus(TransformerMixin, BaseEstimator):
    '''
    Takes in NumPy array with text, and removes underscores and numbers.
    Transform will return NumPy array of cleaned text.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        '''
        df: selects 8k column, cleans text, and outputs corpus
        '''
        for corpus_idx, row in enumerate(X):
            row = row.split()

            for row_idx, word in enumerate(row):
                row[row_idx] = ''.join([c.replace('_','') for c in word
                    if not c.isdigit()])

            X[corpus_idx] = ' '.join(row)

        return X


def score_model(model_name, y_true, y_predicted):
    '''
    Prints precision, recall, and accuracy scores
    Input:
        model_name: string, for display purposes
        y_true: true values of label
        y_predicted: predicted values of label
    '''
    pre = precision_score(y_true, y_predicted)
    rec = recall_score(y_true, y_predicted)
    acc = accuracy_score(y_true, y_predicted)
    print('{0} scores: '.format(model_name))
    print('precision: {0}\nrecall: {1}\naccuracy: {2}'.format(pre, rec, acc))


def draw_profit_curves(profits, model_probs, nb_probs, labels, save=False):
    '''
    Takes predictions for final model, Naive Bayes model and draws profit and
    loss of predictions, and baselines
    Input:
        profits: array of amount gained or lost on each stock price
        model_probs: predicted probability array of final model
        nb_probs: predicted probability array of Naive Bayes model
        labels: true labels of data
        save: set to True to save file
    '''
    # n is the length of all input lists
    n = len(model_probs)
    # generate random predictions for baseline 1
    randos = [np.random.choice([0,1], p =[.48,.52]) for n in range(n)]
    # generate list that always predicts stock price will go up for baseline 2
    longs = [1 for n in range(n)]

    # Use helper method to get profit/loss for models and baselines
    p_ls, cum_p_ls = profit_curve_helper(profits, model_probs, labels)
    # Draw profit/loss for each individual trade of final model
    colors = ['red' if n < 0 else 'blue' for n in p_ls]
    plt.scatter(range(n), p_ls, marker='.', color=colors, alpha=.3)
    plt.xlabel("Individual trades placed, ordered by predicted probability")
    plt.ylabel("Return per trade (percentage)")
    if save:
        plt.savefig('p_l.png')
    else:
        plt.show()
    plt.close()

    # Draw cumulative returns for all models
    plt.plot(range(n), cum_p_ls, color = 'green', label = 'TI + NB')
    p_ls, cum_p_ls = profit_curve_helper(profits, nb_probs, labels)
    plt.plot(range(n), cum_p_ls, color = 'orange', label = 'NB')
    p_ls, cum_p_ls = profit_curve_helper(profits, randos, labels, sort=False)
    plt.plot(range(n), cum_p_ls, color = 'black', label = 'Random')
    p_ls, cum_p_ls = profit_curve_helper(profits, longs, labels, sort=False)
    plt.plot(range(n), cum_p_ls, color = 'blue', label = 'Go long every time')
    plt.xlabel("Cumulative trades placed, ordered by predicted probability")
    plt.ylabel("Cumulative return (percentage)")
    plt.legend()
    if save:
        plt.savefig('profit_curves.png')
    else:
        plt.show()
    plt.close()

def profit_curve_helper(profits, probabilities, labels, sort=True):
    '''
    Takes model probabilities or baseline predictions and calculates the profit
    for correct predictions or loss for incorrect predictions
    '''
    # if providing probabilities, option to sort by high confidence predictions
    if sort:
        # Calculate delta between positive and negative class predicted
        # probabilities, then sort in descending order
        prob_delta = abs(probabilities[:, 0] - probabilities[:, 1])
        prob_delta_index = np.argsort(prob_delta)[::-1]
    else:
        prob_delta_index = range(len(probabilities))
    # p_ls: profit or loss for each trade
    # cum_p_ls: add latest profit or loss to cumulative amount
    p_ls = []
    cum_p_ls = []

    for idx in prob_delta_index:
        if sort:
            predicted = probabilities[idx].argmax()
        else:
            predicted = probabilities[idx]
        true = labels[idx]
        if predicted == true:
            p_ls.append(abs(profits[idx]) * 100)
        else:
            p_ls.append(-abs(profits[idx]) * 100)
        cum_p_ls.append(sum(p_ls))

    return p_ls, cum_p_ls

def run_model(df):
    '''
    Takes in dataframe, and runs model pipelines for Naive Bayes and and final
    models
    Input:
        Pandas data frame with all data
    '''
    # Create pre-label: closing price over opening price after 8-K is filed
    y = df['close'] / df['open']
    # Add column to dataframe to calculate profits later
    df['percent_change'] = (df['close'] - df['open']) / df['open']
    # create positive and negative class (stock go up or stock go down)
    pos_mask = y < 1
    neg_mask = y >= 1
    y = y.where(pos_mask, 1)
    y = y.where(neg_mask, 0)
    y = y.values
    # split dataframe and label into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(df, y,
                                test_size=0.33, random_state=84)

    # grab just 8-K column for Naive Bayes model
    corpus_train = X_train['8k'].values
    corpus_test = X_test['8k'].values

    nlp_pipeline = Pipeline([
        ('prepare', Prepare8kCorpus()),
        ('vectorize', CountVectorizer(
            max_df=.47,
            min_df=18,
            stop_words='english')),
        ('nb', MultinomialNB())])


    nb = nlp_pipeline.fit(corpus_train, y_train)
    nb_train_pred = nb.predict(corpus_train).reshape((corpus_train.shape[0],1))
    nb_test_pred = nlp_pipeline.predict(corpus_test).reshape((corpus_test.shape[0],1))
    nb_test_probs = nlp_pipeline.predict_proba(corpus_test)
    score_model('NB', y_test, nb_test_pred)

    ref = ['ticker','filing_date','prev_close', 'close', 'open', 'percent_change']
    drop = ref + ['Unnamed: 0', '8k']
    ref_train = X_train[ref]
    ref_test = X_test[ref]
    profits = X_test['percent_change'].values
    X_train = X_train.drop(drop, axis=1)
    X_test = X_test.drop(drop, axis=1)

    ti_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ('model', BaggingClassifier(n_estimators=19, n_jobs=-1))
        ])

    ti_alone = ti_pipeline.fit(X_train, y_train)
    ti_test_pred = ti_alone.predict(X_test)
    score_model('TI', y_test, ti_test_pred)
    X_train = np.hstack((nb_train_pred, X_train))
    X_test = np.hstack((nb_test_pred, X_test))

    ti_nlp = ti_pipeline.fit(X_train, y_train)
    ti_nlp_test_pred = ti_pipeline.predict(X_test)
    ti_nlp_test_probs = ti_pipeline.predict_proba(X_test)
    score_model('TI + NLP', y_test, ti_nlp_test_pred)

    draw_profit_curves(profits, ti_nlp_test_probs, nb_test_probs, y_test)

if __name__ == '__main__':
    df = pd.read_csv('data/sp500_big_data_new_feats.csv')
    run_model(df)
