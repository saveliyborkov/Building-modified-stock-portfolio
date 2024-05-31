import pandas as pd
import yfinance as yf
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from datetime import datetime
from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
from sklearn import neighbors


def daterange(start_date, end_date):
    return rrule(DAILY, dtstart=start_date, until=end_date, byweekday=(MO, TU, WE, TH, FR))


def data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end).reset_index()
    df.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": 'volume',
                 'Adj Close': 'adj close'}, inplace=True)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df.date)
    df.sort_values(by='date', inplace=True)
    return df


def markovitz_sharp(data):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    w = ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance()


def indicators(data):
    # momentum
    rsi = ta.momentum.RSIIndicator(data['adj close']).rsi()
    roc = ta.momentum.ROCIndicator(data['adj close']).roc()
    s_o = ta.momentum.StochasticOscillator(data['close'], data['high'], data['low']).stoch()

    # balance
    obv = ta.volume.OnBalanceVolumeIndicator(data['adj close'], data['volume']).on_balance_volume()

    # trend
    macd = ta.trend.MACD(data['adj close']).macd()
    sma = ta.trend.SMAIndicator(data['adj close'], window=50).sma_indicator()

    a = pd.concat([rsi, roc, s_o, obv, macd, sma], axis=1)
    a.dropna(inplace=True)
    return a


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []

    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.concatenate((np.array(dataX), np.array(dataset[0:len(dataset) - time_step - 1, 1]).reshape(-1, 1),
                           np.array(dataset[0:len(dataset) - time_step - 1, 2]).reshape(-1, 1),
                           np.array(dataset[0:len(dataset) - time_step - 1, 3]).reshape(-1, 1)),
                          axis=1), np.array(dataY)


# Котировки с 2012 по 2018, веь 2019 идет ребалансировка каждый месяц
tickers = ['AAPL', 'AMZN', 'HON','JPM', 'NEE','NVDA','SHW', 'TSLA', 'UNH', 'V']
closedate_prediction = ['2018-12-31', '2019-01-31', '2019-02-28', '2019-03-31', '2019-04-30', '2019-05-31',
                        '2019-06-30', '2019-07-31', '2019-08-31', '2019-09-30', '2019-10-31', '2019-11-30',
                        '2019-12-31']


def ml_test(method):
    list3 = []
    datecount = 0
    for m in closedate_prediction[:-1]:
        list1 = []
        for f in tickers:
            df = data(f, '2012-01-01', m)
            ind = indicators(df)
            df = pd.merge(df, ind, left_index=True, right_index=True)
            closedf = df[['date', 'close']]
            close_stock = closedf.copy()
            del closedf['date']
            scaler = MinMaxScaler(feature_range=(0, 1))
            closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

            closedf = np.concatenate((closedf, np.array(df['rsi']).reshape(-1, 1), np.array(df['roc']).reshape(-1, 1),
                                      np.array(df['MACD_12_26']).reshape(-1, 1)), axis=1)

            training_size = int(len(closedf) * 0.9)
            test_size = len(closedf) - training_size
            train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :]

            time_step = 40
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            if method == "SVM":
                model = SVR(kernel='rbf', C=1e2, gamma=0.1)
            if method == 'RF':
                model = RandomForestRegressor(n_estimators=100, random_state=0)
            if method == 'KNN':
                K = time_step
                model = neighbors.KNeighborsRegressor(n_neighbors=K)
            model.fit(X_train, y_train)

            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            #
            train_predict = train_predict.reshape(-1, 1)
            test_predict = test_predict.reshape(-1, 1)

            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            #
            original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
            original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))
            # #
            # # # Evaluation metrices RMSE and MAE
            if m == '2018-12-31':
                method_columns = ['Train', 'Test']
                method_index = ['RMSE', "MSE", "MAE", "Explained var reg score", 'R2', 'MGD', 'MPD']
                method_dict = {'Train': [math.sqrt(mean_squared_error(original_ytrain, train_predict)),
                                         mean_squared_error(original_ytrain, train_predict),
                                         mean_absolute_error(original_ytrain, train_predict),
                                         explained_variance_score(original_ytrain, train_predict),
                                         r2_score(original_ytrain, train_predict),
                                         mean_gamma_deviance(original_ytrain, train_predict),
                                         mean_poisson_deviance(original_ytrain, train_predict)],
                               'Test': [math.sqrt(mean_squared_error(original_ytest, test_predict)),
                                        mean_squared_error(original_ytest, test_predict),
                                        mean_absolute_error(original_ytest, test_predict),
                                        explained_variance_score(original_ytest, test_predict),
                                        r2_score(original_ytest, test_predict),
                                        mean_gamma_deviance(original_ytest, test_predict),
                                        mean_poisson_deviance(original_ytest, test_predict)]}
                method_result_df = pd.DataFrame(method_dict, columns=method_columns, index=method_index)
                method_result_df.to_excel(f+' '+method + '.xlsx')
            x_input1 = test_data[len(test_data) - time_step:, 1:]
            x_input = test_data[len(test_data) - time_step:, :1].reshape(1, -1)
            temp_input = list(x_input)
            temp_input = temp_input[0].tolist()

            temp2_input = pd.DataFrame(x_input1)

            lst_output = []
            n_steps = time_step
            i = 0
            pred_days = 21
            b = 0
            while (i < pred_days):
                temp_input = temp_input + temp2_input.iloc[b].tolist()
                x_input = np.append(x_input, temp2_input.iloc[b].values).reshape(1, -1)
                if (len(temp_input) > time_step + 3):

                    x_input = np.array(temp_input[1:])

                    x_input = x_input.reshape(1, -1)

                    yhat = model.predict(x_input)

                    temp_input = temp_input[:-3]
                    temp_input.extend(yhat.tolist())
                    temp_input = temp_input[1:]

                    lst_output.extend(yhat.tolist())
                    i = i + 1
                    b += 1

                else:
                    yhat = model.predict(x_input)
                    temp_input = temp_input[:-3]

                    temp_input.extend(yhat.tolist())
                    lst_output.extend(yhat.tolist())

                    i = i + 1
                    b += 1
            prediction = scaler.inverse_transform(np.array(lst_output).reshape(1, -1)).tolist()
            prediction = [item for sublist in prediction for item in sublist]
            final_df = df[['date', 'close']].set_index('date')
            final_df.rename(columns={'close': f}, inplace=True)
            index = daterange(datetime.strptime(m, '%Y-%m-%d').date(),
                              datetime.strptime(closedate_prediction[datecount + 1], '%Y-%m-%d').date())
            print(index)
            print(f)
            print(m)
            print(closedate_prediction[datecount + 1])
            print(prediction)
            predicted_df = pd.DataFrame(prediction, columns=[f], index=index[:len(prediction)])
            final_df = pd.concat([final_df, predicted_df])
            list1.append(final_df)
        datecount += 1
        prefinal = pd.concat(list1, axis=1)
        w, d = markovitz_sharp(prefinal)
        df_final = pd.DataFrame(w, index=['2012-01-01 - ' + m])
        list3.append(df_final)
    pd.concat(list3).to_excel(method + 'predict.xlsx')


def market_data():
    tradedata = data(tickers, '2012-01-01', '2019-12-31')[['date', 'close']]
    tradedata.to_excel('maindata2.xlsx')


def markovitz():
    list2 = []
    for b in closedate_prediction[:-1]:
        tradedata_mark = data(tickers, '2012-01-01', b)['close']
        w, d = markovitz_sharp(tradedata_mark)
        df_final = pd.DataFrame(w, index=['2012-01-01 - ' + b])
        df_final['Return'] = d[0]
        df_final['Volatility'] = d[1]
        df_final['Sharp'] = d[2]
        list2.append(df_final)
    pd.concat(list2).to_excel('mark_fin.xlsx')

def markovitz2():
    list2 = []
    for b in closedate_prediction[:-1]:
        tradedata_mark = data(tickers, '2012-01-01', b)['close']
        w, d = markovitz_sharp(tradedata_mark)
        df_final = pd.DataFrame(w, index=['2012-01-01 - ' + b])
        df_final['Return'] = d[0]
        df_final['Volatility'] = d[1]
        df_final['Sharp'] = d[2]
        list2.append(df_final)
    pd.concat(list2).to_excel('mark_fin.xlsx')


def Returns(weights, mean_returns):
    # Annualized portoflio return
    return np.sum(np.multiply(mean_returns, weights))


def STD(weights, cov_matrix):
    # Portoflio annualized standard deviation
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def Sharpe(weights, mean_returns, cov_matrix, risk_free_retrun):
    # sharpe ratio: (mean portfolio return - risk free return)/portfolio standard deviation
    returns = Returns(weights, mean_returns)
    std = STD(weights, cov_matrix)
    return (returns - risk_free_retrun) / std


# market_data()
# markovitz()
# ml_test('SVM')
# ml_test('RF')
# ml_test('KNN')




#
# a = pd.read_excel('mark_fin.xlsx',index_col=None)
# b = pd.read_excel('RFpredict.xlsx',index_col=None)
# df = pd.read_excel('maindata2.xlsx')
#
#
#
# # #
# list_result = []
# for i in range(0,12):
#     means = df[(df['Date'].dt.month == i + 1) & (df['Date'].dt.year == 2019)].set_index('Date')['Close']
#     means = pd.DataFrame(means)
#     print(means)
#     # means = df[(df['date'] <= '2018-12-31')].set_index('date')
#     u = expected_returns.mean_historical_return(means)
#     S = risk_models.sample_cov(means)
#
#     # weights = a.iloc[i,1:11].tolist()
#     # weights2 = b.iloc[i,1:11].tolist()
#     weights = [1]
#     ret1 = sum(np.multiply(u.tolist(), weights))
#     # ret2 = sum(np.multiply(u.tolist(), weights2))
#     vol1 = np.sqrt(np.dot(np.array(weights).T, np.dot(S, weights)))
#     # vol2 = np.sqrt(np.dot(np.array(weights2).T, np.dot(S, weights2)))
#     sharp1 = (ret1 - 0.04) / vol1
#     # sharp2 = (ret2 - 0.04) / vol1
#     list_result.append([ret1,vol1,sharp1])
#     # list_result.append([ret2, vol2, sharp2])
# #
# pd.DataFrame(list_result).to_excel('result.xlsx')

from scipy.stats import ttest_ind

a = pd.read_excel('/Users/saveliyborkov/Documents/Диплом/RESULT_FINAL.xlsx')


v1 = a.iloc[2,2:14].tolist()
v2 = a.iloc[5,2:14].tolist()

res = ttest_ind(v1, v2)

print(res)
