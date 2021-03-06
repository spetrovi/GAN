import random
import numpy as np
import tensorflow as tf


from data_generator import *
from models import *
from functions import *
from utilities import *
from trainers import *

def train(g_model, d_model, a_model, data_generators, epochs=50):
    batch_size = 500
    a_losses = []
    d_losses = []    
    d_accs = []        
    g_mse = []
    for i in range(epochs):
        #Lets use data from different stock every epoch
        data_generator = random.choice(data_generators[1:])
        print('Evaluating Generator')
        g_loss, _, _, _ = g_model.evaluate(data_generators[0].generate_for_gmodel(), steps=1000)
        g_mse.append(g_loss)
        print('Training Discriminator')
        d_batch_loss, d_batch_acc = train_d_better(d_model, data_generator, g_model, epochs=100)
        d_losses += d_batch_loss
        d_accs += d_batch_acc
        print('Training Adversary')             
        for j in range(batch_size):
            
            #Update generator via discriminator's error
            data = next(data_generator.generate_for_amodel())

            a_loss = a_model.train_on_batch(data[0], data[1])
            a_losses.append(a_loss)
            print('>%d, %d/%d, a_loss=%.10f' % (i+1, j+1, batch_size, a_loss))
    print('Finished')       
    return a_losses, d_losses, d_accs, g_mse



#dataset_brent = process_original('data/brent_all_days.csv', ma_days=ma_days)
#dataset_us30 = process_original('data/us30_all_days.csv', ma_days=ma_days)
#dataset_eurusd = process_original('data/eurusd_all_days.csv', ma_days=ma_days)
#dataset_auusd = process_original('data/auusd_all_days.csv', ma_days=ma_days)


#train_percent = 0.9
#train_vol = int(len(dataset)*train_percent)

#train_data = dataset[:train_vol]
#train_data = dataset
#test_data = dataset[train_vol:]

num_steps = 10
batch_size = 10
num_params = 5
data_path = 'saved_models/10d_ma15_fxPro_majors/'
ma_days=15

stocks = glob.glob('data/fxPro_majors/*')
data_generators = []
for stock in stocks:
    dataset = process_fxPro(stock, ma_days)
    data_generators.append(KerasBatchGenerator(dataset, num_steps, batch_size, num_params, skip_step=1))

g_model = define_generator(num_steps)
d_model = define_discriminator(num_steps)
a_model = define_GAN(g_model, d_model)

a_loss, d_loss, d_acc, g_mse = train(g_model, d_model, a_model, data_generators)

save_all(g_model, d_model, data_path)

save_loss(a_loss, data_path+'a_loss')
save_loss(d_loss, data_path+'d_loss')
save_loss(d_acc, data_path+'d_acc')
save_loss(g_mse, data_path+'g_mse')

#data_generator_brent = KerasBatchGenerator(dataset_brent, num_steps, batch_size, num_params, skip_step=1)

#data_generators = [data_generator_brent, data_generator_us30, data_generator_eurusd, data_generator_auusd]

#data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, num_params, skip_step=1)
#test_generator = KerasBatchGenerator(test_data, num_steps, batch_size, num_params, skip_step=1)

#g_model, d_model, a_model = load_all(data_path)
#g_model.evaluate(data_generator.generate_for_gmodel(), steps=1000)


#train_d_better(d_model, data_generator, g_model)
#data_path = 'gan_training/
#data_path = 'gan_training/5'
#save_all(g_model, d_model, 'gan_training20d_mean/')
#predict_recent(g_model, dataset)

#save_for_plot(g_model, dataset, num_steps,'lows_us3020d')
#g_model, d_model, a_model = load_all(data_path)























