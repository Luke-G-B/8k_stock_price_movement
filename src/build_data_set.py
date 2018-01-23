from bs4 import BeautifulSoup
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from collections import defaultdict
import requests
from datetime import datetime, timedelta
from time import sleep, time
import numpy as np
import pandas as pd
import pickle

def get_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, 'lxml')
    company_table = soup.find_all('tr')[1:506]
    company_dict = dict()

    for row in company_table:
        row = row.find_all('td')
        company_dict[row[0].get_text()] = row[7].get_text()

    return company_dict


def get_8k_csv(company_dict, priorto=20180114, count=100):
    data = dict()
    zero_time = time()
    with open('data/8k_data2.csv', 'w') as f:
        f.write('ticker,filing_date,8k\n')

    for ticker, cik in company_dict.items():
        print("Company: {0}, time: {1}".format(ticker, str(time() - zero_time)))
        base_url = 'http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK='\
            +str(cik)+'&type=8-K&dateb='\
            +str(priorto)+'&owner=exclude&output=xml&count='\
            +str(count)
        r = requests.get(base_url)
        data = r.text
        soup = BeautifulSoup(data, 'lxml')
        detail_urls = []
        doc_urls = []

        #Get urls to all detail and document pages for 8-Ks
        for url in soup.find_all('filinghref'):
            # sleep(np.random.randint(3))
            url = url.string
            if url.string.split('.')[len(url.string.split('.')) - 1] == 'htm':
                url+='l'
            detail_urls.append(url)
            # required_url = url[0:-11] # old way
            # required_url += '.txt' # old way
            # doc_urls.append(required_url) # old way

        # get date and time of filing
        filed_dates = []
        corpus = []
        for url in detail_urls:
            # sleep(np.random.randint(3))
            r = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data, 'lxml')
            date_time = soup.find_all('div', {"class" : 'info'})[1].string
            date_time = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
            today = datetime(2018, 1, 18, 0, 0, 0)
            if date_time < today:
                filed_dates.append(date_time)
                # new way to get get text
                url_suffix = soup.find_all('tr')[1].find_all('td')[2].get_text()
                required_url = url[0:-31]
                doc_url = required_url + url_suffix
                doc_urls.append(doc_url) # may not need this
                r = requests.get(doc_url)
                data = r.text
                soup = BeautifulSoup(data, 'lxml')
                text = soup.get_text().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ').replace('\x93', ' ').replace('\x94', ' ').replace(',', '').replace('"','')
                corpus.append(text)
        # getting docs from .txt file
        # for doc in doc_urls:
        #     # sleep(np.random.randint(3))
        #     r = requests.get(url)
        #     data = r.text
        #     soup = BeautifulSoup(data, 'lxml')
        #     text = soup.get_text().replace('\n', ' ').replace('\t', ' ').replace('\xa0', '')
        #     corpus.append(text)
        write_string = ''
        for idx, d in enumerate(filed_dates):
            write_string += ticker +',' + datetime.strftime(d, '%Y-%m-%d %H:%M:%S') + ',' + corpus[idx] + '\n'
        with open('data/8k_data2.csv', 'a') as f:
            f.write(write_string)

    # return {'filing_date':filed_dates, '8-K':corpus}

def get_alpha(ticker_list, alpha_key):
    ts = TimeSeries(key=alpha_key, output_format='pandas')
    ti = TechIndicators(key=alpha_key, output_format='pandas')
    zero_time = time()

    for ticker in ticker_list:
        print('company: {}, time: {}'.format(ticker, str(time() - zero_time)))
        df, meta = ts.get_daily(symbol=ticker, outputsize='full')
        df.to_csv('data/daily/{}.csv'.format(ticker))
        sleep(1)
        df, meta = ti.get_rsi(ticker, interval='daily', time_period=5)
        df.to_csv('data/rsi/{}.csv'.format(ticker))
        sleep(1)
        df, meta = ti.get_cci(ticker, interval='daily', time_period=5)
        df.to_csv('data/cci/{}.csv'.format(ticker))
        sleep(1)
        df, meta = ti.get_obv(ticker, interval='daily')
        df.to_csv('data/obv/{}.csv'.format(ticker))


def build_data_set(csv_file, features):
    zero_time = time()
    df = pd.read_csv(csv_file)
    features_lists = defaultdict(list)

    print('Loading alpha vantage data... time: {}'.format(str(time() - zero_time)))
    for index, row in df.iterrows():
        # need to make sure date is kosher for each ti if row['date']
        date = datetime.strptime(row['filing_date'], '%Y-%m-%d %H:%M:%S')
        ticker = row['ticker'].replace('.','-')
        daily = pd.read_csv('data/daily/{0}.csv'.format(ticker), index_col='date')
        eligible_dates = daily.index
        prev_date = date
        if date.hour >= 16 and date.minute >= 0:
            date += timedelta(days=1)
        else:
            prev_date -= timedelta(days=1)

        while datetime.strftime(date, '%Y-%m-%d') not in set(eligible_dates):
            date += timedelta(days=1)
        while datetime.strftime(prev_date, '%Y-%m-%d') not in set(eligible_dates):
            print(prev_date, ticker)
            prev_date -= timedelta(days=1)
            if datetime.strftime(prev_date, '%Y-%m-%d') < eligible_dates.min():
                prev_date = datetime.strptime(eligible_dates.min(), '%Y-%m-%d')
                break

        date = datetime.strftime(date, '%Y-%m-%d')
        prev_date = datetime.strftime(prev_date, '%Y-%m-%d')
        features_lists['close'].append(daily.loc[date]['4. close'])
        features_lists['open'].append(daily.loc[date]['1. open'])
        features_lists['prev_close'].append(daily.loc[prev_date]['4. close'])

        for feature in features:
            table = pd.read_csv('data/{0}/{1}.csv'.format(feature, ticker), index_col='date')
            if date < table.index.min():
                date = table.index.min()
            datum = table.loc[date]
            features_lists[feature].append(datum)

    for k in features_lists:
        col = np.array(features_lists[k]).reshape((len(features_lists[k]), 1))
        df[k] = col

    return df

def moving_standard_deviation(ticker_list, days=5):
    for ticker in ticker_list:
        daily = pd.read_csv('data/daily/{0}.csv'.format(ticker), index_col='date')
        # get first n days
        std_list = [daily['4. close'].head(5).std() for n in range(5)]
        closing_prices = daily['4. close'].values
        for idx, val in enumerate(closing_prices):
            if idx > 4:
                std_list.append(closing_prices[idx - 5:idx].std())

        std_df = pd.DataFrame(np.array(std_list).reshape(len(std_list), 1))
        std_df = std_df.set_index(daily.index)
        std_df.to_csv('data/moving_std_{0}/{1}.csv'.format(str(days), ticker))

def average_rate_of_change(ticker_list, days=5):
    for ticker in ticker_list:
        daily = pd.read_csv('data/daily/{0}.csv'.format(ticker), index_col='date')
        closing_prices = daily['4. close'].values
        # get the rest of days
        rates = []
        for idx, val in enumerate(closing_prices):
            if idx == 1:
                rates.append((closing_prices[2] - closing_prices[idx]) / closing_prices[idx])
            else:
                rates.append((closing_prices[idx] - closing_prices[idx - 1]) / closing_prices[idx - 1])
        avgs = [rates[idx] for idx in range(days)]
        for idx, val in enumerate(rates):
            if idx >= days:
                avgs.append(np.array(rates[idx - days:idx]).mean())


        df = pd.DataFrame(np.array(avgs).reshape(len(avgs), 1))
        df = df.set_index(daily.index)
        df.to_csv(('data/avg_rate_change_{0}/{1}.csv'.format(str(days), ticker)))


if __name__ == '__main__':

    # company_dict: {ticker:CIK}
    toy_dict = {'MMM':'0000066740', 'ABT':'0000001800', 'AFL':'0000004977',
                    'AGN':'0000884629', 'ALL':'0000899051', 'GOOG': '0001652044',
                    'MO':'0000764180', 'AEP':'0000004904', 'APC':'0000773910',
                    'ADI':'0000006281'}

    d = {'AMG': '0001004434', 'AAL': '0000006201'}

    # big_dict = get_sp500()
    # get_8k_csv(big_dict)
    #
    df = pd.read_csv('data/sp500_8k.csv')
    df2 = pd.read_csv('data/8k_data.csv')
    prev_tickers = df2['ticker'].unique()
    [s.replace('.','-') for s in prev_tickers]
    sp500 = df['ticker'].unique()
    sp500 = [s.replace('.','-') for s in sp500]
    ticker_list = [s for s in sp500 if s not in prev_tickers][204:]

    # download alpha vantage data into folders
    alpha_key = 'ITJ7WH0CRTXEJA4P'
    get_alpha(ticker_list, alpha_key)

    # for loop to get moving std feature
    moving_standard_deviation(ticker_list)
    average_rate_of_change(ticker_list)

    # toy_dict2 = pd.read_pickle('data/toy_8k2.pkl')

    # bring data together and store away in csv
    features = ['rsi', 'obv', 'cci', 'moving_std_5', 'avg_rate_change_5']
    df = build_data_set('data/sp500_8k.csv', features)
    df.to_csv('data/sp500_big_data.csv')
