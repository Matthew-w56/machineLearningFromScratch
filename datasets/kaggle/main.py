import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp


def calculate_error(y_hat, y):
    """Calculates the model's error by averaging each data point's error (y_hat - y) ^ 2"""

    # Square each difference and calculate half the average (half for derivative reasons)
    return np.sum(np.square(y_hat - y)) / (2 * len(y))


def replace_all_dict(_list, _dict, default):
    for i in range(len(_list)):
        result = _dict.get(_list[i])
        if result is not None:
            _list[i] = result
        else:
            _list[i] = default


def replace_all(_list, _item, _replacement):
    for i in range(len(_list)):
        if _list[i] == _item:
            _list[i] = _replacement


def fill_with_0(_list):
    for i in range(len(_list)):
        if not _list[i] > 0:
            _list[i] = 0


file = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

ar = np.array(file)

streetAgainstProperty = ar[:, 3]
isLevel = ar[:, 8]
landSlope = ar[:, 11]
isByRailroad = ar[:, 13:15]
isByRailroadFinished = np.zeros(len(isByRailroad))
overallQuality = ar[:, 17]
overallCondition = ar[:, 18]
monthSold = ar[:, -5]
yearSold = ar[:, -4]
price = ar[:, len(ar[0])-1]

fill_with_0(streetAgainstProperty)
leveldict = dict()
leveldict['Lvl'] = 1
replace_all_dict(isLevel, leveldict, 0)
slopedict = dict()
slopedict['Gtl'] = 0
slopedict['Mod'] = 1
replace_all_dict(landSlope, slopedict, 2)
for i in range(len(isByRailroad)):
    if (isByRailroad[i, 0][0:2] == 'RR') or (isByRailroad[i, 1][0:2] == 'RR'):
        isByRailroadFinished[i] = 1
minYear = np.min(yearSold)
yearSoldFinished = np.zeros(len(yearSold))
for i in range(len(yearSold)):
    yearSoldFinished[i] = yearSold[i] - minYear


data_x = np.column_stack((streetAgainstProperty, isLevel, landSlope, isByRailroadFinished, overallQuality,
                          overallCondition, monthSold, yearSoldFinished))
data_y = np.array(price)



 # TESTESTESTESTESTESTEST ---------------------------------------------------------

file = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

tar = np.array(file)
ids = tar[:, 0]

tstreetAgainstProperty = tar[:, 3]
tisLevel = tar[:, 8]
tlandSlope = tar[:, 11]
tisByRailroad = tar[:, 13:15]
tisByRailroadFinished = np.zeros(len(tisByRailroad))
toverallQuality = tar[:, 17]
toverallCondition = tar[:, 18]
tmonthSold = tar[:, -4]
tyearSold = tar[:, -3]

fill_with_0(tstreetAgainstProperty)
replace_all_dict(tisLevel, leveldict, 0)
replace_all_dict(tlandSlope, slopedict, 2)
for i in range(len(tisByRailroad)):
    if (tisByRailroad[i, 0][0:2] == 'RR') or (tisByRailroad[i, 1][0:2] == 'RR'):
        tisByRailroadFinished[i] = 1
tminYear = np.min(tyearSold)
tyearSoldFinished = np.zeros(len(tyearSold))
for i in range(len(tyearSold)):
    tyearSoldFinished[i] = tyearSold[i] - tminYear


test_x = np.column_stack((tstreetAgainstProperty, tisLevel, tlandSlope, tisByRailroadFinished, toverallQuality,
                          toverallCondition, tmonthSold, tyearSoldFinished))

# ----------------------------------------------------------------------------



model = lm.LinearRegression()

train_x = data_x[:1000, :]
eval_x = data_x[1000:, :]
train_y = data_y[:1000]
eval_y = data_y[1000:]

sc = pp.RobustScaler()

t_train_x = sc.fit_transform(train_x)
t_eval_x = sc.fit_transform(eval_x)

model.fit(t_train_x, train_y)

predictions = model.predict(t_eval_x)

plt.scatter(eval_y, predictions)
plt.plot([0, max(predictions)], [0, max(predictions)], 'r--')
plt.show()

print('Done!')
print(calculate_error(predictions, eval_y))

final_predictions = model.predict(test_x)

output = np.column_stack((ids, final_predictions))
output_file = pd.DataFrame(output)
output_file.to_csv('output.csv')
