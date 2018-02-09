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
    '''
    Goes to Wikipedia to get ticker symbols, and CIK codes for S&P 500 companies
    '''
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, 'lxml')
    company_table = soup.find_all('tr')[1:506]
    company_dict = dict()

    for row in company_table:
        row = row.find_all('td')
        # slice into table to grab ticker and CIK
        company_dict[row[0].get_text()] = row[7].get_text()

    return company_dict


def get_8k_csv(company_dict, priorto=20180114, count=100, filename='data/test.csv'):
    '''
    Input:
        company_dict: this should be a dictionary {symbol:cik}
        priorto: the latest date to include in 8-K filings
        count: number of 8-Ks to show on each page, max 100
        filename: what you want your file called

    Output:
        Writes CSV file with ticker, filing_date, 8-K text
    '''
    data = dict()
    zero_time = time()
    # create file with headers
    with open(filename, 'w') as f:
        f.write('ticker,filing_date,8k\n')

    for ticker, cik in company_dict.items():
        # iterate through each ticker in list
        print("Company: {0}, time: {1}".format(ticker, str(time() - zero_time)))
        base_url = 'http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK='\
            +str(cik)+'&type=8-K&dateb='\
            +str(priorto)+'&owner=exclude&output=xml&count='\
            +str(count)
        r = requests.get(base_url)
        data = r.text
        soup = BeautifulSoup(data, 'lxml')


        detail_urls = []
        # Get urls to all detail pages for 8-Ks
        for url in soup.find_all('filinghref'):
            url = url.string
            # Some urls are 'htm' - need all to be 'html' for slice and dice
            if url.string.split('.')[len(url.string.split('.')) - 1] == 'htm':
                url+='l'
            detail_urls.append(url)

        # get date and time of filing
        filed_dates = []
        corpus = []

        for url in detail_urls:
            r = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data, 'lxml')
            date_time = soup.find_all('div', {"class" : 'info'})[1].string
            date_time = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
            # set today to latest filing date due to prioto date
            # inconsistantly capping max date
            today = datetime(2018, 1, 14, 0, 0, 0)
            if date_time < today:
                filed_dates.append(date_time)
                url_suffix = soup.find_all('tr')[1].find_all('td')[2].get_text()
                # slice off end of detail url, then add url suffix
                required_url = url[0:-31]
                doc_url = required_url + url_suffix
                r = requests.get(doc_url)
                data = r.text
                soup = BeautifulSoup(data, 'lxml')
                # get rid of whitepace, unicode, and punctuation
                text = soup.get_text().replace('\n', ' ').replace('\t', ' ')\
                    .replace('\xa0', ' ').replace('\x93', ' ')\
                    .replace('\x94', ' ').replace(',', '').replace('"','')
                corpus.append(text)

        write_string = ''
        # append line to csv document
        for idx, d in enumerate(filed_dates):
            write_string += ticker +','\
                + datetime.strftime(d, '%Y-%m-%d %H:%M:%S')\
                + ',' + corpus[idx] + '\n'

        with open(filename, 'a') as f:
            f.write(write_string)

def get_alpha(ticker_list, alpha_key):
    '''
    Input:
        ticker_list: list of strings of publicly traded companies symbols, 'XYZ'
        alpha_key: get alpha vantage API key from alpha vantage website
    Output:
        build CSV files of stock price history, and technical indicators for
        each company
    '''

    ts = TimeSeries(key=alpha_key, output_format='pandas')
    ti = TechIndicators(key=alpha_key, output_format='pandas')
    zero_time = time()

    # Build CSV for Daily prices, RSI, CCI, OBV for each company
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


def build_data_set(input_file, output_file, features):
    '''
    Brings together entire data set to be used for modeling
    Input:
        input_file: 8-K file with ticker, filing_date, 8-K text
        features: list of alpha vantage features to add to dataset
    Output:
        returns Pandas dataframe with data to model
    '''
    df = pd.read_csv(input_file)
    features_lists = defaultdict(list)
    # iterate through each row of base dataframe... ticker, filing-date, 8-K
    for index, row in df.iterrows():
        date = datetime.strptime(row['filing_date'], '%Y-%m-%d %H:%M:%S')
        # change '.' to '-' to work with alpha vantage API
        ticker = row['ticker'].replace('.','-')
        daily = pd.read_csv('data/daily/{0}.csv'.format(ticker), index_col='date')
        # make sure filing date is within 2000 - 2018 dates
        eligible_dates = daily.index
        prev_date = date
        # check date... if 8-K filed after market close, change date to next
        # date. If not not, change previous date to one day earlier
        if date.hour >= 16 and date.minute >= 0:
            date += timedelta(days=1)
        else:
            prev_date -= timedelta(days=1)

        # check if date and prev_date are market days. Add or subtract dates,
        # respectively until eligible market day is met
        while datetime.strftime(date, '%Y-%m-%d') not in set(eligible_dates):
            date += timedelta(days=1)
        while datetime.strftime(prev_date, '%Y-%m-%d') not in set(eligible_dates):
            prev_date -= timedelta(days=1)
            # make sure prev_date isn't outside eligible dates
            if datetime.strftime(prev_date, '%Y-%m-%d') < eligible_dates.min():
                prev_date = datetime.strptime(eligible_dates.min(), '%Y-%m-%d')
                break

        # set close to close after 8-K filed
        # set open to open after 8-K filed
        # set prev_close to close before 8-K filed
        date = datetime.strftime(date, '%Y-%m-%d')
        prev_date = datetime.strftime(prev_date, '%Y-%m-%d')
        features_lists['close'].append(daily.loc[date]['4. close'])
        features_lists['open'].append(daily.loc[date]['1. open'])
        features_lists['prev_close'].append(daily.loc[prev_date]['4. close'])

        # add each feature to dataframe from csv
        for feature in features:
            table = pd.read_csv('data/{0}/{1}.csv'.format(feature, ticker), index_col='date')
            print (date, prev_date, feature, ticker)
            if prev_date < table.index.min():
                prev_date = table.index.min()
            datum = table.loc[prev_date]
            features_lists[feature].append(datum)

    for feat in features_lists:
        col = np.array(features_lists[feat]).reshape((len(features_lists[feat]), 1))
        df[feat] = col

    df.to_csv(output_file)
    return df

def moving_standard_deviation(ticker_list, days=5):
    '''
    calculate moving standard deviation on closing prices, frequency in days
    '''
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


def x_rate_of_change(ticker_list, features, column=None, days=5):
    for ticker in ticker_list:

        for feature in features:
            df = pd.read_csv('data/{0}/{1}.csv'.format(feature, ticker), index_col='date')
            if column == None:
                col = df.iloc[:, 0].values
            else:
                col = df[column].values
            # get the rest of days
            rates = []

            for idx, val in enumerate(col):
                if col[idx] == 0:
                    col[idx] = .00001
                if idx == 1:
                    rates.append((col[2] - col[idx]) / col[idx])
                else:
                    rates.append((col[idx] - col[idx - 1]) / col[idx - 1])

            avgs = [rates[idx] for idx in range(days)]

            for idx, val in enumerate(rates):
                if idx >= days:
                    avgs.append(np.array(rates[idx - days:idx]).mean())

            rate_df = pd.DataFrame(np.array(avgs).reshape(len(avgs), 1))
            rate_df = rate_df.set_index(df.index)
            rate_df.to_csv(('data/{0}_avg_rate_change_{1}/{2}.csv'.format(feature, str(days), ticker)))

def ema_ratio(ticker_list, slow=10, fast=5):
    ti = TechIndicators(key=alpha_key, output_format='pandas')
    for ticker in ticker_list:
        print(ticker)
        slow_df, meta = ti.get_ema(ticker, interval='daily', time_period=slow)
        fast_df, meta = ti.get_ema(ticker, interval='daily', time_period=fast)
        ratio_df = slow_df.EMA / fast_df.EMA
        ratio_df.fillna(1, inplace=True)
        ratio_df = ratio_df.rename('{0}_{1}_mar'.format(str(slow), str(fast)))
        # ratio_df = ratio_df.set_index(df.index)
        ratio_df.to_csv('data/{0}_{1}_mar/{2}.csv'.format(str(slow), str(fast), ticker), header=True)

if __name__ == '__main__':
    # for timing
    zero_time = time()
    # company_dict: {ticker:CIK}
    small_dict = {'MMM':'0000066740', 'ABT':'0000001800', 'AFL':'0000004977',
                    'AGN':'0000884629', 'ALL':'0000899051', 'GOOG': '0001652044',
                    'MO':'0000764180', 'AEP':'0000004904', 'APC':'0000773910',
                    'ADI':'0000006281'}


    big_dict = get_sp500()
    print('Scraping 8-Ks: {}'.format(str(time() - zero_time)))
    # get_8k_csv(small_dict)

    df = pd.read_csv('data/test.csv')
    tickers = df['ticker'].unique()
    tickers = [s.replace('.','-') for s in tickers]


    # download alpha vantage data into folders
    print('Downloading stock price info: {}'.format(str(time() - zero_time)))
    alpha_key = '' # get key from alpha vantage
    get_alpha(tickers, alpha_key)

    print('Engineering new features: {}'.format(str(time() - zero_time)))
    # for loop to get moving std feature
    moving_standard_deviation(tickers)
    average_rate_of_change(tickers)
    ema_ratio(tickers, slow=10, fast=5)
    ema_ratio(tickers, slow=20, fast=10)
    print('Writing all data to csv...: {}'.format(str(time() - zero_time)))
    # bring data together and store away in csv
    rate_features = ['rsi', 'obv', 'cci']
    x_rate_of_change(tickers, rate_features)
    features = ['rsi', 'obv', 'cci', 'moving_std_5', 'avg_rate_change_5','rsi_avg_rate_change_5', 'obv_avg_rate_change_5', 'cci_avg_rate_change_5', '10_5_mar', '20_10_mar']
    df = build_data_set('data/test.csv', 'data/test_full.csv', features)
