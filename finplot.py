import matplotlib.pyplot as plt
import json
import numpy as np
import csv
from data_generator import process_fxPro
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from functions import *

def real_prediction(g_model, dataset, num_days, num_par7ams):
    batchsize = 30
    x = np.zeros((batchsize, num_days, num_params))
    y = np.zeros((batchsize, num_days + 1, num_params))
    open_real = []
    close_real = []
    high_real = []
    low_real = []
    
    for i in range(batchsize):
        index = len(dataset) - batchsize - num_days + i 
        x_days = dataset[index : index + num_days]
        y_days = dataset[index : index + num_days + 1]
        x[i, :] = x_days.copy()
        y[i, :] = y_days.copy()
        open_real.append(y_days[-1][0])
        high_real.append(y_days[-1][1])
        low_real.append(y_days[-1][2])
        close_real.append(y_days[-1][3])
    

    mean = np.mean(x)
    std = np.std(x)

    x = (x - mean) / std
    y = (y - mean) / std

    
    prediction = g_model.predict(x)

    prediction = mean + (prediction * std)

    open_pred = []
    close_pred = []
    low_pred = []
    high_pred = []
    for day in prediction:
        open_pred.append(day[-1][0])
        high_pred.append(day[-1][1])
        low_pred.append(day[-1][2])
        close_pred.append(day[-1][3])
           
    dates = np.arange(batchsize)
    
    day_offset = .05
    OC_offset = .4
    fig, ax = plt.subplots()
    ax.vlines(dates - day_offset, low_real, high_real, color='blue')
    ax.hlines(open_real, dates - OC_offset - day_offset , dates - day_offset, color='blue')
    ax.hlines(close_real, dates + OC_offset- day_offset , dates - day_offset, color='blue')
    
    ax.vlines(dates + day_offset, low_pred, high_pred, color='green')
    ax.hlines(open_pred, dates - OC_offset + day_offset , dates + day_offset, color='green')
    ax.hlines(close_pred, dates + OC_offset + day_offset, dates + day_offset, color='green')
       
    ax.autoscale_view()
    ax.grid()
#    plt.show()
    plt.savefig('sample.png')
    
model_path = './saved_models/2022-04-27-23-05-42_GAN'
g_model = tf.keras.models.load_model(model_path+'/g_model', compile=False)
opt = Adam(lr=0.009, beta_1=0.9)
g_model.compile(loss=my_gMSE, optimizer='adam', metrics=[my_gMSE, my_gMSE_o_c, my_mDir])       

stock = '/home/spetrovi/git/GAN/data/fxPro_majors/#USSPX500_Daily_201106140000_202103190000.csv'
ma_days = 7
name, dataset = process_fxPro(stock, ma_days)
num_days = 5
num_params = 5


real_prediction(g_model, dataset, num_days, num_params)

#dates, opens, highs, lows, closes, volumes = list(map(list, zip(*dataset)))
#dates = np.array(dates)



#offset = .4

#Hight Low Lines


#Open Lines

#ax.hlines(opens, dates - offset, dates)

#Close Lines

#ax.hlines(closes, dates + offset, dates)


#plt.savefig('sample.png')


