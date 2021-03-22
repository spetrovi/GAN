import numpy as np
import random
import tensorflow as tf
from models import *

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
    
def save_for_plot(g_model, dataset, numsteps, name):
    lows_real = []
    lows_pred = []
    for i in range(numsteps, len(dataset)-1):
        recent_data = np.array([dataset[i-numsteps : i]])

        mean = np.mean(recent_data)
        std = np.std(recent_data)

        #normalise
        norm_r = (recent_data-mean)/std
        prediction = g_model.predict(norm_r)
    
        #denormalise
        prediction = mean + (prediction[0][-1] * std)
    
        # dataset[0:5] > indexes 0..4, next value is at dataset[5]
        lows_real.append(dataset[i][2])
        lows_pred.append(prediction[2])

    #write to file for later analysis
    w = open(name, 'w')
    for val in lows_real:
        w.write(str(val) + ',')
    w.write('\n')
    for val in lows_pred:
        w.write(str(val) + ',')
    w.close()
    
def save_loss(losses, name='loss'):    
    w = open(name, 'w')
    for val in losses:
        w.write(str(val) + ',')
    w.close()


def predict_recent(g_model, dataset):
    recent_data = np.array([dataset[len(dataset)-5 : len(dataset)]])
    mean = np.mean(recent_data)
    std = np.std(recent_data)

    #normalise
    norm_r = (recent_data-mean)/std


    prediction = g_model.predict(norm_r)
    print('Actual day:')
    print(recent_data)

    print('Predicted next day:')
    print(mean + (prediction*std))
    
def save_all(g_model, d_model, name):
    g_model.save(name + 'g_model')
    d_model.save(name + 'd_model')

def load_all(name):
    g_model = tf.keras.models.load_model(name + 'g_model', compile=False)
    d_model = tf.keras.models.load_model(name + 'd_model', compile=False)
    a_model = define_GAN(g_model, d_model)
    
    g_model.compile(loss=my_gMSE, optimizer='adam',  metrics=[my_gMSE, my_gMAE_l, my_gMAE_h])
    d_model.compile(loss=my_dloss, optimizer='adam',  metrics=[my_dacc])
    
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
