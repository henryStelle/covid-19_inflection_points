import numpy as np
import scipy.optimize as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b * t))


def getLogisticCoef(data):
    x = np.arange(len(data['total_cases']))
    y = np.array(data['total_cases'])
    smoothedY = []
    (a, b, c), cov = optim.curve_fit(logistic, x, y, bounds=(0, [10000000., 5., 100000000000.]), p0=(1, 0.1, 1),
                                     maxfev=1000000)
    for t in x:
        smoothedY.append(logistic(t, a, b, c))
    secondDerv = np.diff(smoothedY, n=2)
    inflectionDate = None
    for i, v in enumerate(secondDerv.tolist()):
        if v < 0:
            inflectionDate = allData.columns[4 + i]
            break
    return {'a': a, 'b': b, 'c': c, 'inflectionDate': inflectionDate, 'inflectionHeight': logistic(i, a, b, c),
            'smoothed': smoothedY, 'rSquared': r2_score(data['total_cases'].tolist(), smoothedY)}


print('Program Initialized')
print('Downloading Data...')
allData = pd.read_csv("https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv")
print('Download Complete')

while True:
    lookupCounty = input('\nWhat county would you like to look up? Leave blank for all.\n')
    lookupState = input('\nWhat state (abbr) would you like to look up? Leave blank for all.\n')
    for i, row in allData.iterrows():
        county = row['County Name']
        state = row['State']
        cases = row.tolist()[4:]
        if lookupCounty.lower() in str(county).lower() and lookupState.lower() in str(state).lower():
            res = getLogisticCoef(pd.DataFrame(cases, columns=['total_cases']))
            plt.figure(num='COVID-19 Trends')
            plt.plot(np.array(list(allData.columns[4:])), np.array(cases), '--', label="Cases")
            plt.plot(np.array(list(allData.columns[4:])), np.array(res['smoothed']), label='Logistic Curve ($R^2$ = {:.2f})'.format(res['rSquared']))
            if res['inflectionDate'] is not None:
                print('\n{}, {} passed their inflection point on {}'.format(county, state, res['inflectionDate']))
                plt.plot(res['inflectionDate'], res['inflectionHeight'], 'ro', label='Inflection Point')
                plt.text(res['inflectionDate'], res['inflectionHeight'],
                         "({}, {:.2f})  ".format(res['inflectionDate'], res['inflectionHeight']),
                         horizontalalignment='right',
                         verticalalignment='top', )
            else:
                print('\n{}, {} has not yet passed their inflection point'.format(county, state))
            plt.xlabel('Date')
            plt.ylabel('Total Cases')
            plt.legend()
            plt.title("{}, {}".format(county, state))
            show = input('Show graph (y/n)? Enter "stop" to exit this loop.\n')
            if show.lower() == 'y':
                plt.show()
            elif show.lower() == 'stop':
                plt.close()
                break
            else:
                plt.close()