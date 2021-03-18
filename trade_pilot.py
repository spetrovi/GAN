import random
import numpy as np
import tensorflow as tf


from data_generator import *
from models import *
from functions import *
from utilities import *
from trainers import *

def train(g_model, d_model, a_model, data_generator, test_generator, epochs=50):
    batch_size = 100
    a_losses = []
    d_losses = []    
    for i in range(epochs):
        print('Pretraining Discriminator')
        d_batch_loss = train_d_better(d_model, data_generator, g_model, epochs=100)
        d_losses += d_batch_loss
        print('Evaluating generator')   
        g_model.evaluate(test_generator.generate_for_gmodel(), steps=100)
        print('Training Adversary')                
        for j in range(batch_size):
            
            #Update generator via discriminator's error
            data = next(data_generator.generate_for_amodel())

            a_loss = a_model.train_on_batch(data[0], data[1])
            a_losses.append(a_loss)
            print('>%d, %d/%d, a_loss=%.10f' % (i+1, j+1, batch_size, a_loss))
    print('Finished')       
    return a_losses, d_losses


#dataset = process_original('eurusd_all_days.csv', 'eurusd_daily_processed.csv')
dataset = process_original('brent_all_days.csv')
train_percent = 0.9
train_vol = int(len(dataset)*train_percent)

#train_data = dataset[:train_vol]
train_data = dataset
test_data = dataset[train_vol:]

num_steps = 5
batch_size = 10
num_params = 5
latent_dim = 100
data_path = 'gan_training/'

g_model = define_generator()
d_model = define_discriminator()
a_model = define_GAN(g_model, d_model)

data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, num_params, latent_dim, skip_step=1)
test_generator = KerasBatchGenerator(test_data, num_steps, batch_size, num_params, latent_dim, skip_step=1)



a_loss, d_loss = train(g_model, d_model, a_model, data_generator, test_generator)


save_loss(a_loss, 'losses/a_loss')
save_loss(d_loss, 'losses/d_loss')
#train_d_better(d_model, data_generator, g_model)

save_all(g_model, d_model, data_path)
predict_recent(g_model, dataset)

#save_for_plot(g_model, dataset, 'overtrained')
#g_model, d_model, a_model = load_all(data_path)























