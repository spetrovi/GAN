import numpy as np
import random
import tensorflow as tf
from models import *
from data_generator import *

from functions import *
def shuffle_inputs(X, y):

    X_new = np.zeros((len(X), len(X[0]), len(X[0][0])))
    y_new = []

    
    positions = [i for i in range(len(X))]

    random.shuffle(positions)

    for i, pos in enumerate(positions):
        X_new[i,:] = X[pos]
        y_new.append(y[pos])
        
        
    return X_new, np.array(y_new)
    
#Normalise prices to [0,1] interval
def normalise(price, high, low):
    return (price - low) / (high - low)

#86.716 15.965
def denormalise(price, high=86.716, low=15.965):
    return price * (high - low) + low
 
def analyse(g_model, dataset, numsteps, name):
    batchsize = 250    
    num_params = 9

    current_idx = 0

    result = {'open_real':[], 'open_pred':[], 'close_real':[], 'close_pred':[]
    }
    for b in range(len(dataset)//batchsize):
        x = np.zeros((batchsize, numsteps, num_params))
        for i in range(batchsize):
            x_days = dataset[current_idx : current_idx + numsteps]
            y_days = dataset[current_idx : current_idx + numsteps + 1]
            x[i, :] = x_days.copy()                    
            result['open_real'].append(y_days[-1][0])
            result['close_real'].append(y_days[-1][3])

            current_idx += 1

        mean = np.mean(x)
        std = np.std(x)
        
        x = (x - mean) / std
        _min = np.abs(np.min(x))
        
        x = x + _min

        prediction = g_model.predict(x)
        prediction = mean + ((prediction - _min) * std)

#        print(prediction)
        
        for seq in prediction:
            result['open_pred'].append(seq[-1][0])
            result['close_pred'].append(seq[-1][3])

    signal_correct = 0
    signal_incorrect = 0
    all_days = len(result['open_pred'])
    correct_pips = 0
    incorrect_pips = 0
    for i in range(all_days):
        if (result['open_real'][i] < result['close_real'][i]) and (result['open_pred'][i] < result['close_pred'][i]):
            signal_correct += 1
            correct_pips += result['close_real'][i] - result['open_real'][i]
        elif (result['open_real'][i] > result['close_real'][i]) and (result['open_pred'][i] > result['close_pred'][i]):
            signal_correct += 1
            correct_pips += result['open_real'][i] - result['close_real'][i]
        else:
            signal_incorrect += 1
            incorrect_pips += abs(result['open_real'][i] - result['close_real'][i])
    print('Correct, all, ratio', signal_correct, all_days, signal_correct/all_days)
    print('Winning pips: ', correct_pips)
    print('Losing pips: ', incorrect_pips)
    print('Profit pips: ', int(correct_pips - incorrect_pips))

def save_for_plot(g_model, dataset, numsteps, name):
    lows_real = []
    lows_pred = []
    high_real = []
    high_pred = []
    batchsize = 250    
    current_idx = 0
    num_params = 9
    
    for b in range(len(dataset)//batchsize):
        x = np.zeros((batchsize, numsteps, num_params))
        for i in range(batchsize):
            x_days = dataset[current_idx : current_idx + numsteps]
            y_days = dataset[current_idx : current_idx + numsteps + 1]
            x[i, :] = x_days.copy()                    
            high_real.append(y_days[-1][1])
            lows_real.append(y_days[-1][2])

            current_idx += 1

        mean = np.mean(x)
        std = np.std(x)
        
        x = (x - mean) / std
        _min = np.abs(np.min(x))
        
        x = x + _min

        prediction = g_model.predict(x)
        prediction = mean + ((prediction - _min) * std)

#        print(prediction)
        
        for seq in prediction:
            lows_pred.append(seq[-1][1])
            high_pred.append(seq[-1][2])

        
        
    #write to file for later analysis
    w = open(name, 'w')
    for val in lows_real:
        w.write(str(val) + ',')
    w.write('\n')
    for val in lows_pred:
        w.write(str(val) + ',')
    w.write('\n')
    for val in high_real:
        w.write(str(val) + ',')
    w.write('\n')
    for val in high_pred:
        w.write(str(val) + ',')
    w.close()
    
    
    
def save_loss(losses, name='loss'):    
    w = open(name, 'w')
    for val in losses:
        w.write(str(val) + ',')
    w.close()


def predict_recent(g_model, dataset, numsteps):
    recent_data = np.array([dataset[len(dataset)-numsteps : len(dataset)]])
    mean = np.mean(recent_data)
    std = np.std(recent_data)

    #normalise
    norm_r = (recent_data-mean)/std


    prediction = g_model.predict(norm_r)
    print('Actual day:')
    print(recent_data)

    print('Predicted next day:')
    print(mean + (prediction*std))
    
def save_all(g_model, d_model, a_model, name):
    g_model.save(name + 'g_model')
    d_model.save(name + 'd_model')
    d_model.save(name + 'a_model')    

def load_all(name):
    g_model = tf.keras.models.load_model(name + 'g_model', compile=False)
    d_model = tf.keras.models.load_model(name + 'd_model', compile=False)
    a_model = define_GAN(g_model, d_model)
    
    g_model.compile(loss=my_gMSE, optimizer='adam',  metrics=[my_gMSE, my_gMAE_l, my_gMAE_h])
    
    opt = Adam(lr=0.006, beta_1=0.9)   
    d_model.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
    
    return g_model, d_model, a_model
    
def make_predictions(g_model, path_to_stocks, ma_days, num_steps, batch_size, num_params):
    stocks = glob.glob(path_to_stocks + '/*')
    
    for stock in stocks:
        dataset = process_fxPro(stock, ma_days)
        recent_data = np.array([dataset[len(dataset)-num_steps : len(dataset)]])
        
        generator = KerasBatchGenerator(dataset, num_steps, batch_size, num_params, skip_step=1)
        _, _, g_MSE_lows, g_MSE_highs = g_model.evaluate(generator.generate_for_gmodel(), steps=100)
        
        mean = np.mean(recent_data)
        std = np.std(recent_data)

        #normalise
        norm_r = (recent_data-mean)/std        
        prediction = g_model.predict(norm_r)
        print('Prediction for stock: ' + stock)
        print('Previous days:')
        print(recent_data)

        print('Next day:')
        print(mean + (prediction*std))
        
        print('Error for Lows: ' + str(g_MSE_lows*std))        
        print('Error for Highs: ' + str(g_MSE_highs*std))        

        print('\n')
