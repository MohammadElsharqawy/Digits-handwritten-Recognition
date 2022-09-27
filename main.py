import numpy as np
import pandas as pd

data = pd.read_csv('data.csv', header=None)  # data is a datarame table

#print(data.shape, data)



D0 = data.copy()
D1 = data.copy()
D2 = data.copy()
D3 = data.copy()
D4 = data.copy()
D5 = data.copy()
D6 = data.copy()
D7 = data.copy()
D8 = data.copy()
D9 = data.copy()

datasets = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]  # list of dataframes

for i in range(10):
    for j in range(5000):
        if datasets[i].at[j, 400] == i:
            datasets[i].at[j, 400] = 1
        else:
            datasets[i].at[j, 400] = 0

# TrainTest_datadict['key'][0] ==> X,  input pixes(features)
# TrainTest_datadict['key'][1] ==> y,  output

TrainTest_datadict = {}

for i in range(10):
    X = datasets[i].iloc[:, :400]  # from 0 to 399, 400 col
    X = X.T  # 3lshan n3mlo dot fe gradient descent 3ltol (400*5000)

    y = datasets[i].iloc[:, -1]  # last vector (output)

    y = np.array([y]) #[[]]  (1*5000)
    TrainTest_datadict['D' + str(i)] = [X, y]

#print(TrainTest_datadict['D0'][0])
#print(TrainTest_datadict['D0'][1])

itterations = 10  # random big value
alpha = 0.160
m = 5000
cost_values = []
trained_parameters = []

for trainsets in range(10):
    X = TrainTest_datadict['D' + str(trainsets)][0] #(400*5000)
    y = TrainTest_datadict['D' + str(trainsets)][1]  #(1*5000)
    weights = np.random.randn(1, 400)  # random values with shape 1*400

    bias = 0
    costfunc_values = []  # cost for each model will be putted in cost_values
    k = 0
    print('Training for dataset ' + str(trainsets))
    for i in range(1, itterations + 1):
        # logistic function
        z = np.dot(weights, X) + bias  #(1*400 . 400*5000)->> (1,5000)
        hypothesis = 1 / (1 + np.exp(-z)) #(1,5000)
        #print(hypothesis.shape)
        # cost function
        j = 1 / m * (-1 * (np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)))) #scalar

        costfunc_values.append(j)
        k += 1

        # gradient decent
        dw = 1 / m * np.dot(hypothesis - y, X.T)  #(1*5000)*(5000*400)
        db = 1 / m * np.sum(hypothesis - y) #scalar
        weights = weights - alpha * dw #1*400
        bias = bias - alpha * db

        # stop training

        if i >= 3 and abs(j - costfunc_values[-2]) < 0.000001 and abs(j - costfunc_values[-3]) < 0.000001:
            break
    cost_values.append(costfunc_values)
    trained_parameters.append([weights, bias])
    print('itteration number:', k)

# calculate accuracy for each classifier
for datasetnum in range(10):
    X = TrainTest_datadict['D' + str(datasetnum)][0] #400*5000
    y = TrainTest_datadict['D' + str(datasetnum)][1]  #1*5000
    weights = trained_parameters[datasetnum][0] #1*400
    bias = trained_parameters[datasetnum][1]
    correct_predictions = 0
    for i in range(5000):
        ##inputs.iloc[:,i].T single image
        z = np.dot(weights, X.T.iloc[i, :].T) + bias ##fit the line with the jth photo #1*400 * 400*1
        hypothesis = 1 / (1 + np.exp(-z)) #1*1
        if (hypothesis >= 0.5) & y[0, i] == 1: # if predict == 1 and it is 1 that correct
            correct_predictions += 1
        if np.logical_and(hypothesis < 0.5, y[0, i] == 0):
            correct_predictions += 1
            # print(correct_predictions)
    acc = (correct_predictions / 5000) * 100
    print('accuracy for dataset ' + str(datasetnum), " = ", acc)



#1 V All classfier calculation accuracy


inputs = data.iloc[:, :400]
inputs = inputs.T #(400*5000)
outputs = data.iloc[:, -1]



#outputs = np.array(outputs)


accuratepredicts = 0
for i in range(5000):
    probabilities = []
    for j in range(10):
        weights = trained_parameters[j][0]
        bias = trained_parameters[j][1]
    ##inputs.iloc[:,i] single image
    #400*1
    #weight 1*400
        z = np.dot(weights, inputs.iloc[:, i]) + bias ## fit the line with the jth photo
        hypothesis = 1/(1 + np.exp(-z))
        probabilities.append(hypothesis)
    predict = probabilities.index(max(probabilities))
    if outputs[i] == predict:
        accuratepredicts += 1

print((accuratepredicts/5000) *100)