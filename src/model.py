import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc

# class NLP(TransformerMixin, BaseEstimator):
#
#     def __init__(self):
#         pass
#
#     def fit(self, X, y):
#         '''
#         X: corpus of text to process
#         y: rate (ie. open / previous close)
#         '''
#         self.X = X
#         pos_mask = y < 1
#         neg_mask = y >= 1
#         self.y = y.where(pos_mask, 1)
#         self.y = y.where(neg_mask, 0)
#
#         # Remove numerals from corpus
#         for corpus_idx, row in enumerate(self.X):
#             row = row.split()
#             for row_idx, word in enumerate(row):
#                 row[row_idx] = ''.join([c for c in word if not c.isdigit()])
#             self.X[corpus_idx] = ' '.join(row)
#
#
#
#         return self



def nlp(df):
    # create y label
    y = df['close'] / df['open'] # actionable
    df['percent_change'] = (df['close'] - df['open']) / df['open'] # actionable
    # y = df['open'] / df['prev_close'] # knowlege gain
    # df['percent_change'] = (df['open'] - df['prev_close']) / df['prev_close'] # knowledge gain

    # pos or neg
    pos_mask = y < 1
    neg_mask = y >= 1
    y = y.where(pos_mask, 1)
    y = y.where(neg_mask, 0)

    # clean corpus more
    corpus = df['8k'].values
    for corpus_idx, row in enumerate(corpus):
        row = row.split()
        for row_idx, word in enumerate(row):
            row[row_idx] = ''.join([c.replace('_','') for c in word if not c.isdigit()])
        corpus[corpus_idx] = ' '.join(row)
    df['8k'] = corpus

    #split
    X_train, X_test, y_train, y_test = train_test_split(df, y,
                                test_size=0.33, random_state=42)

    corpus_train = X_train['8k'].values
    corpus_test = X_test['8k'].values

    nb = MultinomialNB()

    tf_vectorizer = CountVectorizer(max_df=.47, min_df=18,
                                        stop_words='english')

    tf_train = tf_vectorizer.fit_transform(corpus_train).toarray()
    tf_test = tf_vectorizer.transform(corpus_test).toarray()

    nb.fit(tf_train, y_train)
    nb_test_pred = nb.predict(tf_test).reshape((tf_test.shape[0],1))
    nb_train_pred = nb.predict(tf_train).reshape((tf_train.shape[0],1))
    nb_test_probs = nb.predict_proba(tf_test)

    pre = precision_score(y_test, nb_test_pred)
    rec = recall_score(y_test, nb_test_pred)
    acc = accuracy_score(y_test, nb_test_pred)
    print('NLP alone: ')
    print('precision: {0}\nrecall: {1}\naccuracy: {2}'.format(pre, rec, acc))

    ref_df_train = X_train[['ticker','filing_date','prev_close', 'close', 'open', 'percent_change']]
    ref_df_test = X_test[['ticker','filing_date', 'prev_close', 'close', 'open', 'percent_change']]


    # combine with rest of features
    profits = X_test.pop('percent_change')
    X_train = X_train.drop(['Unnamed: 0', 'filing_date', 'ticker', '8k', 'prev_close', 'close', 'open', 'percent_change'], axis=1)
    X_test = X_test.drop(['Unnamed: 0', 'filing_date', 'ticker', '8k', 'prev_close', 'close', 'open'], axis=1)


    #scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.hstack((nb_train_pred, X_train))
    X_test = np.hstack((nb_test_pred, X_test))

    models = {'Ada':AdaBoostClassifier(), 'Bagging':BaggingClassifier(),
        'extra':ExtraTreesClassifier(), 'gradient': GradientBoostingClassifier()}#,
        # 'linearSVC': LinearSVC(), 'SVC':SVC(probability=True)}
    # for m in range(17,30):
    models = {'Bagging':BaggingClassifier(n_estimators=19, n_jobs=-1), 'rf':RandomForestClassifier(n_estimators=19)}


    for m in models:
        model = models[m]
        model.fit(X_train, y_train)
        print('non-NLP: for {}'.format(m), model.score(X_test, y_test))

        # multi-class
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('combined score for {}: '.format(m))
        pre = precision_score(y_test, y_pred)#, labels=['buy', 'short'], average='micro')
        rec = recall_score(y_test, y_pred)#, labels=['buy', 'short'], average='micro')
        acc = accuracy_score(y_test, y_pred)
        print('precision: {0}\nrecall: {1}\naccuracy: {2}'.format(pre, rec, acc))

    return y_pred, model.predict_proba(X_test), y_test.values, profits.values, nb_test_probs

def roc(probabilities, labels):
    thresholds = np.sort(probabilities)
    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold

        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)

        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives

        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)

        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()

def accuracy_curve(probabilities, labels):
    positive_probs = probabilities[:, 1]
    thresholds = np.sort(positive_probs)
    accuracy_rate, inaccuracy_rate = [], []

    # num_positive_cases = sum(labels)
    # num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predictions = [0 if p < threshold else 1 for p in positive_probs]

        # Calculate the number of correctly predicted cases
        acc = accuracy_score(labels, predictions)
        accuracy_rate.append(acc)
        inaccuracy_rate.append(1 - acc)

    plt.plot(thresholds, accuracy_rate)
    plt.xlabel("Probabilities")
    plt.ylabel("Accuracy Rate")
    # plt.savefig('accuracy_curve_new_feats.png')
    plt.show()
    plt.close()


def profit_curve(profits, probabilities, labels, nb_probs):
    prob_delta = abs(probabilities[:, 0] - probabilities[:, 1])
    prob_delta_index = np.argsort(prob_delta)[::-1]

    p_ls = []
    cum_p_ls = []

    for idx in prob_delta_index:
        predicted = probabilities[idx].argmax()
        true = labels[idx]
        if predicted == true:
            p_ls.append(abs(profits[idx]) * 100)
        else:
            p_ls.append(-abs(profits[idx]) * 100)
        cum_p_ls.append(sum(p_ls))

    nb_p_ls =[]
    nb_cum_p_ls = []
    prob_delta = abs(nb_probs[:, 0] - nb_probs[:, 1])
    prob_delta_index = np.argsort(prob_delta)[::-1]

    for idx in prob_delta_index:
        predicted = nb_probs[idx].argmax()
        true = labels[idx]
        if predicted == true:
            nb_p_ls.append(abs(profits[idx]) * 100)
        else:
            nb_p_ls.append(-abs(profits[idx]) * 100)
        nb_cum_p_ls.append(sum(nb_p_ls))

    randos = [np.random.choice([0,1], p =[.48,.52]) for n in range(len(cum_p_ls))]
    rando_p_ls = []
    rando_cum_p_ls = []

    for idx, predicted in enumerate(randos):
        true = labels[idx]
        if predicted == true:
            rando_p_ls.append(abs(profits[idx]) * 100)
        else:
            rando_p_ls.append(-abs(profits[idx]) * 100)
        rando_cum_p_ls.append(sum(rando_p_ls))

    longs = [1 for n in range(len(cum_p_ls))]
    long_p_ls = []
    long_cum_p_ls = []

    for idx, predicted in enumerate(longs):
        true = labels[idx]
        if predicted == true:
            long_p_ls.append(abs(profits[idx]) * 100)
        else:
            long_p_ls.append(-abs(profits[idx]) * 100)
        long_cum_p_ls.append(sum(long_p_ls))


    colors = ['red' if n < 0 else 'blue' for n in p_ls]
    plt.scatter(range(len(cum_p_ls)), p_ls, marker='.', color=colors, alpha=.3)
    plt.xlabel("Trades")
    plt.ylabel("Return per trade")
    # plt.savefig('p_l_new_feats.png')
    plt.show()
    plt.close()

    plt.plot(range(len(cum_p_ls)), cum_p_ls, color = 'blue', label = 'TI + NB')
    plt.plot(range(len(nb_cum_p_ls)), nb_cum_p_ls, color = 'orange', label='NB')
    plt.plot(range(len(cum_p_ls)), rando_cum_p_ls, color='green', label='Random')
    plt.plot(range(len(cum_p_ls)), long_cum_p_ls, color='black', label='Go long every time')
    plt.xlabel("Number of trades placed")
    plt.ylabel("Cumulative return (percentage)")
    plt.legend()
    plt.show()
    # plt.savefig('profit_curve_new_feats.png')
    plt.close()

    return p_ls


def save_fig(tprs, fprs, thresholds, probabilities):
    area_under_curve = auc(fprs, tprs)
    print("AUC: {}".format(auc))

    plt.plot(fprs, tprs)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC plot of Long")
    plt.show()
    # plt.savefig('ROC_plot_of_Long_big_data_new_feats.png')
    plt.close()

    return area_under_curve




def rmse(y_predicted, y_true):
    return (((y_true - y_predicted) ** 2).sum() / len(y_true)) ** 0.5


if __name__ == '__main__':
    df = pd.read_csv('data/sp500_big_data_new_feats.csv')
    # # features = ['rsi', 'obv', 'cci', 'moving_std']
    # # X = df[features].values
    # # y = df['close'] #/ df['prev_close']
    # # X_train, X_test, y_train, y_test = train_test_split(
    # #     X, y, test_size=0.33, random_state=123)
    # # scaler = StandardScaler()
    # # X_train_trans = scaler.fit_transform(X_train)
    # # X_test_trans = scaler.transform(X_test)
    preds, probs, true, profits, nb_probs = nlp(df)
    # probabilities_pos = probs[:,1]
    # tprs, fprs, thresholds = roc(probabilities_pos, true)
    # area_under_curve = save_fig(tprs, fprs, thresholds, probabilities_pos)
    # print('AUC: ', area_under_curve)
    profit_curve(profits, probs, true, nb_probs)
    # accuracy_curve(probs, true)
