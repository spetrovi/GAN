import csv
import glob
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Layer, LeakyReLU, Lambda, LSTM, TimeDistributed, Dropout, Flatten, RepeatVector
from tensorflow.keras.optimizers import Adam

#from keras.utils import *
import tensorflow as tf

import random

import keras.backend as K
import statistics 
# train the discriminator model
def train_discriminator(model, data_generator, g_model, n_iter=1000, n_batch=256):
    half_batch = int(n_batch / 2)
    #manualy enumerate epochos
    for i in range(n_iter):
        #get a batch of real data
        #true_data_generator = KerasBatchGenerator(dataset, num_steps, batch_size, latent_dim, num_params, skip_step=num_steps)
        data = next(data_generator.generate_for_dmodel())
        X_real, y_real = data[0], data[1]
        
        #update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
	    
        #generate 'fake' examples
        data = next(data_generator.generate_pred(g_model))
        X_fake, y_fake = data[0], data[1]
        y_fake = np.zeros((len(y_fake)))
        #update discriminator on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
 
        #summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
        
def shuffle_inputs(X, y):

    X_new = np.zeros((len(X), len(X[0]), len(X[0][0])))
    y_new = []

    
    positions = [i for i in range(len(X))]

    random.shuffle(positions)

    for i, pos in enumerate(positions):
        X_new[i,:] = X[pos]
        y_new.append(y[pos])
        
        
    return X_new, np.array(y_new)


# train the discriminator model
def train_d_better(model, data_generator, g_model, n_iter=1000, n_batch=256):
    half_batch = int(n_batch / 2)
    #manualy enumerate epochos
    for i in range(n_iter):
        #get a batch of real data
        #true_data_generator = KerasBatchGenerator(dataset, num_steps, batch_size, latent_dim, num_params, skip_step=num_steps)
        data = next(data_generator.generate_for_dmodel(trues=True))
        X_real, y_real = data[0], data[1]

        #generate 'fake' examples
        data = next(data_generator.generate_pred(g_model))
        X_fake, y_fake = data[0], data[1]

        #update discriminator on fake samples

        X = np.concatenate((X_real, X_fake), axis=0)
        y = np.concatenate((y_real, y_fake), axis=0)

        X, y = shuffle_inputs(X, y)

        loss, acc = model.train_on_batch(X, y)
 
        #summarize performance
        print('>%d d_loss=%.10f, acc=%.0f%%' % (i+1, loss, acc*100))        

#y is in this format
#It is a batch of ys we generate in generator
#the y is values for 6 days and array with prediction (one value copied) as 7th element
#For standalone dloss function, we only need predictions
#The paper does the Dloss as follows
#-1/m * sum(log(sigmoid when inputing real)) - 1/m * sum(log(1-sigmoid when inputig fake))
def my_dloss(y_true,y_pred):
    
    #from every element of batch, elem, we want to take elem[-1][0]
    #for this we need to do some transpositions, to select correct data
    y_trues = tf.transpose(y_true, perm=[1,0,2])[-1]
    y_trues = tf.transpose(y_trues)[0]
    
    y_preds = tf.transpose(y_pred, perm=[1,0,2])[-1]
    y_preds = tf.transpose(y_preds)[0]    
    
    #Now we need to separate the predictions
    #Xreal will contain predictions for when data was real
    #Xfake will contain predictions for when data was fake

    #get indexes of truthfull y
    eqs = tf.math.equal(y_trues, 1)
    idx = tf.where(eqs)
    
    #predictions for when data was real
    Xreal = tf.gather(y_preds, idx)

    log_real = tf.math.log(Xreal)
    sum_real = tf.math.reduce_sum(log_real)
    
    fin_real = tf.math.multiply(sum_real, -(1/y_true.shape[0]))
    
    
    #get indexes of fake y
    eqs = tf.math.equal(y_trues, 0)
    idx = tf.where(eqs)
    
    #predictions for when data was fake
    Xfake = tf.gather(y_preds, idx)

    sub_fake = tf.math.subtract(1.0, Xfake)

    log_fake = tf.math.log(sub_fake)
    sum_fake = tf.math.reduce_sum(log_fake)
    
    fin_fake = tf.math.multiply(sum_fake, 1/y_true.shape[0])

    return tf.math.subtract(fin_real, fin_fake)
    
def my_aloss(y_true,y_pred):
    
    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-2]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-2]
    
    subs = tf.math.subtract(predx1, realx1)
    squares = tf.math.multiply(subs, subs)
    
    xsum = tf.math.reduce_sum(squares)
    gMSE = tf.math.multiply(xsum, 1/y_true.shape[0])

    #gloss
    y_preds = tf.transpose(y_pred, perm=[1,0,2])[-1]
    Xfake = tf.transpose(y_preds)[0]    

    sub_fake = tf.math.subtract(1.0, Xfake)
    log_fake = tf.math.log(sub_fake)
    sum_fake = tf.math.reduce_sum(log_fake)    
    gloss = tf.math.multiply(sum_fake, 1/y_true.shape[0])
    return gMSE - gloss


def my_mse(y_true,y_pred):
    y_true = tf.transpose(y_true)[-1]
    y_pred = tf.transpose(y_pred)[-1]
    return K.mean(K.square(y_pred-y_true))
    
# train the generator model
#g_model input = 5 days (X)
#g_model output = 6 days (y)
def train_g_better(g_model, data_generator, n_iter=10, n_batch=256):
    half_batch = int(n_batch / 2)
    #g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])    
#    g_model.fit(data_generator.generate_for_gmodel(), steps_per_epoch = 100, batch_size=10, epochs=100)
    #manualy enumerate epochos
    for i in range(n_iter):

        #get a batch of  data
        (X, y) = next(data_generator.generate_for_gmodel())
    
    
        loss, mse, mae, mape = g_model.train_on_batch(X, y)
       #To check on values by generator
        data = next(data_generator.generate_pred(g_model))
        #summarize performance
        print('>%d g_loss=%.10f, mse=%.10f , mae=%.10f, mape=%.10f' % (i+1, loss, mse, mae, mape))        
        
        
#Normalise prices to [0,1] interval
def normalise(price, high, low):
    return (price - low) / (high - low)

#86.716 15.965
def denormalise(price, high=86.716, low=15.965):
    return price * (high - low) + low
    
    
def process_original(name, output, ma_days=5):
    orig = glob.glob(name)[0]
    bars = []
    closes = []
    with open(orig, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(csvreader):
            if i > 0 and float(row[5]) > 0.0:           
                open_p = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                
                if i == 1:
                    max_p = open_p
                    min_p = open_p
                
                
                if i > ma_days:
                    #moving average for past X days
                    s = 0
                    for j in range(1, 1+ma_days):
                        s += closes[-j]
                    moving_avg = s / ma_days
                    bars.append(np.array([open_p,high,low,close,moving_avg]))
                    
                    #For normalisation purposes
                    if max([open_p,high,low,close]) > max_p:
                        max_p = max([open_p,high,low,close])
                    if min([open_p,high,low,close]) < min_p:
                        min_p = min([open_p,high,low,close])
                closes.append(close)
    out = []
    print(max_p, min_p)
    for l in bars:
        out.append(list(map(lambda x: x, l)))

    return np.array(out)
	
class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, num_params, latent_dim, skip_step=1):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_params = num_params
        self.latent_dim = latent_dim
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate_for_dmodel(self, trues=True):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.ones((self.batch_size, self.num_steps+2, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]
                mean = np.mean(x_days)
                std = np.std(x_days)

                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))
                    

                x[i, :] = norm_days
                
                if trues:
                    norm_days.append(np.ones((self.num_params)))
                else:
                    norm_days.append(np.zeros((self.num_params)))
                    
                y[i, :] = norm_days
                
                self.current_idx += self.skip_step

            yield (x, y)
            
    def generate_for_amodel(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.ones((self.batch_size, self.num_steps+2, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                mean = np.mean(x_days)
                std = np.std(x_days)

                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))
                    

                x[i, :] = norm_days
                
                
                x_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]
                mean = np.mean(x_days)
                std = np.std(x_days)

                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))

                norm_days.append(np.zeros((self.num_params)))

                y[i, :] = norm_days
                
                self.current_idx += self.skip_step

            yield (x, y)            

            
    def generate_fake(self):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.zeros((self.batch_size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    
                x[i, :] =[[random.uniform(0, 1) for k in range(self.num_params)] for l in range(self.num_steps+1)]
                self.current_idx += self.skip_step

            yield (x, y)
            

    #Generate training data for generator model
    #X is batches of 5 days (inputs)
    #y are the same days but with one more day (outputs)
    def generate_for_gmodel(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                y_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]
                
                mean = np.mean(y_days)
                std = np.std(y_days)
                
                x_norm_days = []
                y_norm_days = []
                for d in x_days:
                    x_norm_days.append(list((d - mean) / std))
                    
                for d in y_days:
                    y_norm_days.append(list((d - mean) / std))                    
                x[i, :] = x_norm_days
                y[i, :] = y_norm_days
                self.current_idx += self.skip_step

            yield (x, y)
            
    #Generate for training d model
    def generate_pred(self, g_model):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.zeros((self.batch_size, self.num_steps+2, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                mean = np.mean(x_days)
                std = np.std(x_days)
#                predict_day = g_model.predict(np.array(([self.data[self.current_idx:self.current_idx + self.num_steps]])) )[0][-1]
                predict_day = g_model.predict(np.array(([(x_days - mean) / std])))
#                print('SOURCE DATA')
#                print(np.array(([x_days])))
#                print('Predicting')                
#                print( (predict_day * std) + mean) 
                predict_day = predict_day[0][-1]
                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))     

                x[i, :] = np.append(norm_days, [predict_day], 0)
                
                norm_days.append(self.data[self.current_idx + self.num_steps + 1])
                norm_days.append(np.zeros((self.num_params)))
                y[i, :] = norm_days
                
                
                self.current_idx += self.skip_step
 
            yield (x, y)        
            
    def generate_eval(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.zeros((self.batch_size, self.num_steps+1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]                
                y[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps+1]

                self.current_idx += self.skip_step

            yield (x, y)



def define_generator():    
    num_steps = 5
    num_params = 5
    A1 = Input(shape=(num_steps,num_params), name='A1')
    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), name='A2')(A1)
    A21 = LeakyReLU(name='A21')(A2)     
    A3 = LSTM(num_params, return_sequences=False, input_shape=(num_steps,num_params), name='A3')(A21)
    A31 = LeakyReLU(name='A31')(A3) 
    A32 = Dropout(0.5, name='A32')(A31)
#    A4 = TimeDistributed(Dense(1, activation='sigmoid'), name='A4')(A3)
#    A4 = TimeDistributed(Dense(5, activation='relu'), name='A4')(A3)
    A5 = Layer(Dense(1), name='A5')(A32)
    A51 = LeakyReLU(name='A51')(A5) 
    A6 = tf.keras.layers.Reshape((1,5))(A51)
    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A6])

    merged = Model(inputs=[A1], outputs=[C])
    return merged
    

#define the standalone discriminator model
#in order to perfect loss function, we need to pass input to output and concatenate with prediction
#input is list of 6 days, we need to concatenate it wih 6 times discriminator prediction
#we need to use functional model for this
def define_discriminator(num_steps=6, num_params=5):
    
    A1 = Input(shape=(num_steps,num_params), name='A1')
    A2 = Dense(72, name='A2')(A1)
    A21 = LeakyReLU(name='A21')(A2)   
    A3 = Dense(100, name='A3')(A21)
    A31 = LeakyReLU(name='A31')(A3)
    A4 = Dense(10, name='A4')(A31)
    A41 = LeakyReLU(name='A41')(A4)
    
    #Ok, here it gets messy, so I'll explain a bit
    #We have some computation going on with, but our input shape is num_steps*num_params
    #That means, that each node's output is array of num_steps elements
    #But for prediction, we only want one!
    #So we use Flatten. This takes the outputs and make a single straight array, that we densely connect to single neuron
    A42 = Flatten(input_shape=(num_steps, 1))(A41)
    A5 = Dense(1, activation='sigmoid', name='A5')(A42)
    #Ok, we have our predicted value in A5, however, we can't glue it to the original values,
    #Tensorflow wants element of array to have the same dimmension
    #Our input array, that we want to pass is num_steps arrays of num_params elements
    #In order to glue our prediction, we need to make a fake num_params array
    #To make it a num_params array, we can use RepeatVector to just repeat the predicted value
    #Don't forget, later in loss function, we need to make the prediction back to single value    
    A51 = RepeatVector(num_params)(A5)

    #Reshaping, cause tf is stupid (or too smart actually)
    A6 = tf.keras.layers.Reshape((1,num_params))(A51)

    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A6])
    
    merged = Model(inputs=[A1], outputs=[C])
    merged.compile(loss=my_dloss, optimizer='adam',  metrics=['accuracy'])
    return merged


    
#define GAN by sequentialy combining generator and discriminator
#We freeze the discriminator model, so it is updated only when we train it alone
#When we train GAN, we want to update only generator
def define_GAN(g_model, d_model):
    #make weights in discriminator not trainable
    d_model.trainable = False
    #connect
    model = Sequential()
    #add generator
    model.add(g_model)
    #add discriminator
    model.add(d_model)
    #compile
    #maybe this Adam settings is not good
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=my_aloss, optimizer=opt)
    return model




def train(g_model, d_model, a_model, data_generator, test_generator, n_epochs=50, n_batch=256):
    batch_size = 1000
    for i in range(n_epochs):
        print('Pretraining Discriminator')
        train_d_better(d_model, data_generator, g_model, n_iter=50)            
        
        print('Evaluating generator')
        g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])       
        g_model.evaluate(test_generator.generate_for_gmodel(), steps=100)
  
        print('Training Adversary')                
        for j in range(batch_size):
            #Train discriminator with true and fake data            
            #Generate real examples
#            data = next(data_generator.generate_for_dmodel())
#            X_real, y_real = data[0], data[1]

            #generate 'fake' examples
#           data = next(data_generator.generate_pred(g_model))
#            X_fake, y_fake = data[0], data[1]
#            y_fake = np.zeros((len(y_fake)))

            #Merge and shuffle
#            X = np.concatenate((X_real, X_fake), axis=0)
 #           y = np.concatenate((y_real, y_fake), axis=0)
    
  #          X, y = shuffle_inputs(X, y)
            
            #update discriminator on fake samples
#            d_loss, d_acc = d_model.train_on_batch(X, y)
            #Maybe it could be better to pretrain discriminator much more
              
            
            #Update generator via discriminator's error
            data = next(data_generator.generate_for_amodel())

            a_loss = a_model.train_on_batch(data[0], data[1])
            print('>%d, %d/%d, a_loss=%.10f' % (i+1, j+1, batch_size, a_loss))







#dataset = process_original('eurusd_all_days.csv', 'eurusd_daily_processed.csv')
dataset = process_original('brent_all_days.csv', 'brent_daily_processed.csv')
train_percent = 0.9
train_vol = int(len(dataset)*train_percent)

#train_data = dataset[:train_vol]
train_data = dataset
test_data = dataset[train_vol:]

num_steps = 5
batch_size = 10
num_params = 5
latent_dim = 100

g_model = define_generator()
d_model = define_discriminator()
a_model = define_GAN(g_model, d_model)

data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, num_params, latent_dim, skip_step=1)
test_generator = KerasBatchGenerator(test_data, num_steps, batch_size, num_params, latent_dim, skip_step=1)
train(g_model, d_model, a_model, data_generator, test_generator)

#g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])       

#train_g_better(g_model, data_generator)
#train_d_better(d_model, test_generator, g_model)

"""     
#Pretrain generator and discriminator for a bit
g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])       
train_g_better(g_model, data_generator, n_iter=50)
train_d_better(d_model, data_generator, g_model, n_iter=500)
#Train GAN
train(g_model, d_model, a_model, data_generator, test_generator)


#save the generator model
data_path_g = "gan_training/1_gen"
data_path_d = "gan_training/1_dis"
g_model.save(data_path_g)
d_model.save(data_path_d)
"""
#g_model = tf.keras.models.load_model(data_path_g, compile=False)
#d_model = tf.keras.models.load_model(data_path_d, compile=False)
#load the model
#g_model = tf.keras.models.load_model(data_path_g, compile=False)

#train discriminator on a good generator and save it
#g_model = tf.keras.models.load_model(data_path_g, compile=False)
#g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])       
#train_d_better(d_model, data_generator, g_model)


#d_model.save(data_path_d)
#d_model = tf.keras.models.load_model(data_path_d, compile=False)
#d_model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
#train_d_better(d_model, data_generator, g_model)
#a_model = define_GAN(g_model, d_model)

#train(g_model, d_model, a_model, data_generator, test_generator)
#g_model.save("gan_proper/2_generator")
#d_model.save("gan_proper/2_discriminator")

def save_for_plot(g_model, dataset):
    lows_real = []
    lows_pred = []
    for i in range(5, len(dataset)-1):
    #for i in range(5, 20):
        recent_data = np.array([dataset[i-5 : i]])

        mean = np.mean(recent_data)
        std = np.std(recent_data)

        #normalise
        norm_r = (recent_data-mean)/std
        prediction = g_model.predict(norm_r)
    
        #denormalise
        prediction = mean + (prediction[0][-1] * std)
    
        lows_real.append(dataset[i+1][2])
        lows_pred.append(prediction[2])

    #write to file for later analysis
    w = open('lows', 'w')
    for val in lows_real:
        w.write(str(val) + ',')
    w.write('\n')
    for val in lows_pred:
        w.write(str(val) + ',')
    w.close()



"""
#g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])       
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
"""

























