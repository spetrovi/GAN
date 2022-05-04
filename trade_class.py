import random
import json
import os
import models
import numpy as np
import glob
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from utilities import analyse, real_prediction
from data_generator import process_fxPro, norm_MeanStd, KerasBatchGenerator
from functions import my_dacc, my_dloss,my_gMSE,my_gMSE_o_c, my_mDir,my_gMAE, openMAE, highMAE, lowMAE, closeMAE, nextOpenMAE
from tensorflow.keras.optimizers import Adam

def normalise(y_days):
    features = np.transpose(y_days)
    price_features = np.vstack((features[0], features[1], features[2], features[3], features[-2], features[-1]))
    mean_price = np.mean(price_features)
    std_price = np.std(price_features)
    y_days = (price_features - mean_price) / std_price
            
    mean_tick = np.mean(features[4])
    std_tick = np.std(features[4])
    y_days = np.vstack((y_days, (features[4] - mean_tick) / std_tick))
            
    y_days = np.transpose(y_days)
    
    return y_days, mean_price, std_price
class GAN():
    def __init__(self, g_model, d_model, num_steps, num_params, ma_days, batch_size, opt):
        self.g_model = g_model
        self.d_model = d_model
        self.a_model = models.define_GAN(g_model, d_model, opt)
        self.num_steps = num_steps
        self.num_params = num_params
        self.ma_days = ma_days
        self.batch_size = batch_size
        self.result_path = './saved_models/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_GAN/'
        os.makedirs(self.result_path)
        self.data_generators = []
        self.data_sets = []
        self.get_generators()
        
    def train_GAN(self, gan_epochs=15, gan_batch_size=200):
        a_losses = []
        d_losses = []    
        g_mse = []
        for i in range(gan_epochs):
            #Lets use data from different stock every epoch
            data_generator = random.choice(self.data_generators)
            print('Evaluating Generator')
            g_loss, _, _, _, _, _, _ = self.g_model.evaluate(data_generator.generate_for_gmodel(), steps=50)
            g_mse.append(g_loss)
            print('Training Discriminator')
            data_generator = random.choice(self.data_generators)
            d_batch_loss, d_batch_acc = self.train_discriminator(data_generator, epochs=100)
            #discriminator too good, we ned to increase adversary training
            if np.mean(d_batch_acc) > 0.9:
                gan_batch_size = 300
            if np.mean(d_batch_acc) < 0.6:
                gan_batch_size = 200            
            d_losses += d_batch_loss
            print('Training Adversary')
            for j in range(gan_batch_size):
                #Update generator via discriminator's error
                data_generator = random.choice(self.data_generators)
                X, y = next(data_generator.generate_for_amodel())
                a_loss = self.a_model.train_on_batch(X, y)
                a_losses.append(a_loss)
                print('>%d, %d/%d, a_loss=%.10f' % (i+1, j+1, gan_batch_size, a_loss))
        print('Finished')
        self.plot_stats(a_losses, 'a_loss')
        self.plot_stats(d_losses, 'd_loss')
        self.plot_stats(g_mse, 'g_mae')      
#        self.plot_stats(g_mse_oc_l, 'g_mse_oc')
#        self.plot_stats(my_mDir_l, 'direction')  
    
    def train_discriminator(self, data_generator, epochs=100):
        d_losses = []
        d_accs = []
        self.d_model.trainable = True
        #manualy enumerate epochos
        for i in range(epochs):
            #get a batch of real data
            #X_real -> [batch_size][num_steps+1, e.g. days][num_params]
            #y_real -> The same, but every batch element has a row of ones
            X_real, y_real = next(data_generator.generate_for_dmodel_real())
        
            #generate fake examples, i.e. sequence of num_steps days + generated day
            X_fake, y_fake = next(data_generator.generate_for_dmodel_fake(self.g_model))

            #we merge the real and faked data
            X = np.concatenate((X_real, X_fake), axis=0)
            y = np.concatenate((y_real, y_fake), axis=0)
    
            #shuffle the input
            X, y = self.shuffle_inputs(X, y)

            loss, acc = self.d_model.train_on_batch(X, y)
            d_losses.append(loss)
            d_accs.append(acc)
            
            #summarize performance
            print('>Epoch: %d d_loss=%.10f, acc=%.0f%%' % (i+1, loss, acc*100))
        self.d_model.trainable = False        
        return d_losses, d_accs
        
    def get_generators(self):
#       stock_paths = glob.glob('./data/forex_since2020/*') + glob.glob('./data/majors_since2020/*')
       stock_paths = glob.glob('./data/majors_since2020/*')
       for stock in stock_paths:
           name, dataset = process_fxPro(stock, self.ma_days)
           self.data_generators.append(KerasBatchGenerator(dataset, num_steps, batch_size, num_params, skip_step=1, name=name))
           self.data_sets.append((name, dataset))

    def save_models(self):
        self.g_model.save(self.result_path + 'g_model')
        self.d_model.save(self.result_path + 'd_model')
        self.a_model.save(self.result_path + 'a_model')

    def plot_stats(self, data, name):
        plt.plot(data, label=name)
        plt.grid()
        plt.legend()
        plt.savefig(self.result_path + name + '.svg', dpi=100)
        plt.close()

    #We want to shuffle, but in a way that y mirrors X
    def shuffle_inputs(self, X, y):
        X_new = np.zeros((len(X), len(X[0]), len(X[0][0])))
        y_new = []

        positions = [i for i in range(len(X))]
        random.shuffle(positions)
        
        for i, pos in enumerate(positions):
            X_new[i,:] = X[pos]
            y_new.append(y[pos])        
        return X_new, np.array(y_new)    
    
    def g_evaulation(self):
            data_generator = self.data_generators[0]
            #print('Evaluating Generator')
            g_loss, _, my_gMSE_o_c, my_mDir= self.g_model.evaluate(data_generator.generate_for_gmodel(), steps=100)
            return g_loss
#        for name, dataset in self.data_sets:
#            real_prediction(self.g_model, dataset, self.num_steps, name, self.num_params)

    def all_stocks(self):
        for name, dataset in self.data_sets:
            self.real_prediction(name, dataset)

    def real_prediction(self, name, dataset, time_size = 30):
        batchsize = time_size
        num_days = self.num_steps
        x = np.zeros((batchsize, num_days, self.num_params))
        y = np.zeros((batchsize, num_days + 1, self.num_params))
        open_real = []
        close_real = []
        high_real = []
        low_real = []
        ma5_real = []
        futureopen_real = []
        means = []
        stds = []
    
        index = random.randint(0,len(dataset) - batchsize - num_days)
        for i in range(batchsize):
            index +=1
            y_days = dataset[index : index + num_days + 1]

            open_real.append(y_days[-1][0])
            high_real.append(y_days[-1][1])
            low_real.append(y_days[-1][2])
            close_real.append(y_days[-1][3])

            
            #Normalisation process
            x_days, mean_price, std_price = norm_MeanStd(y_days[:-1])
            means.append(mean_price)
            stds.append(std_price)
            #End of normalisation
            
            x[i, :] = x_days.copy()

        prediction = g_model.predict(x)
        d_prediction = d_model.predict(prediction)

        open_pred = []
        close_pred = []
        low_pred = []
        high_pred = []
        futureopen_pred = []
        disc_pred = []
        
        for i, day in enumerate(prediction):
            #denorm process
            features = np.transpose(day)
            price_features = np.vstack((features[0], features[1], features[2], features[3], features[4], features[5]))
            day = np.transpose((price_features * stds[i]) + means[i])

            open_pred.append(day[-1][0])
            high_pred.append(day[-1][1])
            low_pred.append(day[-1][2])
            close_pred.append(day[-1][3])
            futureopen_pred.append(day[-1][5])
            disc_pred.append(d_prediction[i][-1][0])
           
        dates = np.arange(batchsize)
    
        day_offset = .05
        OC_offset = .4
        fig, ax = plt.subplots()
        ax.vlines(dates - day_offset, low_real, high_real, color='blue')
        ax.hlines(open_real, dates - OC_offset - day_offset , dates - day_offset, color='blue')
        ax.hlines(close_real, dates + OC_offset- day_offset , dates - day_offset, color='blue')
    
        categories = (np.array(disc_pred) > 0.65).astype(int)

        colormap = np.array(['yellow', 'green'])
    
        ax.vlines(dates + day_offset, low_pred, high_pred, color='green')#colormap[categories])
        ax.hlines(open_pred, dates - OC_offset + day_offset , dates + day_offset, color='green')#colormap[categories])
        ax.hlines(close_pred, dates + OC_offset + day_offset, dates + day_offset, color='green')#colormap[categories])
        
        futureopen_pred = [futureopen_pred[-1]] + futureopen_pred[:-1]
        
#        ax.scatter(dates, futureopen_pred,color='red')#colormap[categories])
        
        ax.autoscale_view()
        ax.grid()
        plt.savefig(self.result_path+name.split('/')[-1]+'_chart.png')
        
        #TEST strategy
        signal_correct = 0
        signal_incorrect = 0
        profit_pips = 0
        all_days = len(open_pred)
        #New strategy
        #IF mean of predicted day is above mean of the last X days
        #   Buy on Open and exit on Close
        #   If incorrect, take -50 profit 
#        for i in range(all_days - 1 -1):
 #           mean_of_predicted = low_pred[i+1] + ((high_pred[i+1] - low_pred[i+1]) / 2)
  #          if ma5_real[i] < mean_of_predicted:
#                print('Ma5 Closes: ',ma5_real[i])
 #               print('Mean of predicted: ', mean_of_predicted)
  #              print('Next day: ', open_real[i+1], high_real[i+1], low_real[i+1], close_real[i+1])            
   #             if close_real[i+2] > open_real[i+1]:
   #                 print('Good guess!!!')
    #                signal_correct += 1
     #               profit_pips += close_real[i+2] - open_real[i+1]
      #          else: #actual price went lower
       #             signal_incorrect += 1
        #            if open_real[i+1] - close_real[i+2] > 50:                    
          #              profit_pips -= 50
         #           else:
           #             profit_pips -= open_real[i+1] - close_real[i+2]
    #                print('Bad guess!!!')
            #if ma5_real[i] > mean_of_predicted:
#                print('Ma5 Closes: ',ma5_real[i])
 #               print('Mean of predicted: ', mean_of_predicted)
  #              print('Next day: ', open_real[i+1], high_real[i+1], low_real[i+1], close_real[i+1])            
             #   if close_real[i+2] < open_real[i+1]:
   #                 print('Good guess!!!')
              #      signal_correct += 1
               #     profit_pips +=  open_real[i+1] - close_real[i+2]
                #else: #actual price went lower
                 #   signal_incorrect += 1
                  #  if open_real[i+1] - close_real[i+2] < 50:                    
                  #      profit_pips -= 50
                  #  else:
                   #     profit_pips -= close_real[i+2] - open_real[i+1]
    #                print('Bad guess!!!')                        
            
        print(name)
        print('Correct/Incorrect: ', signal_correct, signal_incorrect)
        print('Profit pips: ', int(profit_pips))

#       Try predict future half
        #open_future = []
        #close_future = []
        #low_future = []
        #high_future = []
        #new_x = [x[time_size//2:(time_size//2)+1]]

        #for i in range(time_size //2 ):
           # prediction = g_model.predict(new_x)[0]
          #  new_x = np.zeros((1, num_days, self.num_params))
         #   new_x[0,:] = prediction[1:]
        #    prediction = means[time_size//2] + (prediction * stds[time_size//2])
       #     open_future.append(prediction[-1][0])
      #      high_future.append(prediction[-1][1])
     #       low_future.append(prediction[-1][2])
    #        close_future.append(prediction[-1][3])            

   #     ax.vlines(dates[time_size//2:] + day_offset, low_future, high_future, color='red')
  #      ax.hlines(open_future, dates[time_size//2:] - OC_offset+ day_offset  , dates[time_size//2:]+ day_offset , color='red')
 #       ax.hlines(close_future, dates[time_size//2:] + OC_offset + day_offset, dates[time_size//2:] + day_offset, color='red')
#        plt.savefig(self.result_path+name.split('/')[-1]+'_chart.png')
        
num_steps = 5
batch_size = 100
num_params = 7
ma_days = 7
opt = Adam(lr=0.001, beta_1=0.999)    



#d_model = models.discriminator_orig(num_steps, num_params, opt)

#dobra dvojka
#g_model = models.define_generator(num_steps, num_params, opt)
#d_model = models.define_discriminator_LSTM(num_steps, num_params, opt)

#g_model = models.generator_orig(num_steps, num_params, opt)
#d_model = models.define_discriminator_LSTM(num_steps, num_params, opt)

path = '2022-05-03-22-40-09_GAN'
g_model = tf.keras.models.load_model('./saved_models/'+path+'/g_model', compile=False)
d_model = tf.keras.models.load_model('./saved_models/'+path+'/d_model', compile=False)
d_model.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
g_model.compile(loss=my_gMAE, optimizer=opt, metrics=[my_gMAE, openMAE, highMAE, lowMAE, closeMAE, nextOpenMAE]) 

my_gan = GAN(g_model, d_model, num_steps, num_params, ma_days, batch_size, opt)

my_gan.train_GAN(gan_epochs=100)
my_gan.save_models()
my_gan.all_stocks()
#########a = my_gan.g_evaulation()
########3print(a)












