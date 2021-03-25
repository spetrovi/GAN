import numpy as np
from utilities import *

# train the discriminator model
def train_d_better(model, data_generator, g_model, epochs=1000):
    d_losses = []
    d_accs = []
    model.trainable = True
    #manualy enumerate epochos
    for i in range(epochs):
        #get a batch of real data
        #true_data_generator = KerasBatchGenerator(dataset, num_steps, batch_size, latent_dim, num_params, skip_step=num_steps)
        data = next(data_generator.generate_for_dmodel_real())
        X_real, y_real = data[0], data[1]

        #generate 'fake' examples
        data = next(data_generator.generate_for_dmodel_fake(g_model))
        X_fake, y_fake = data[0], data[1]

        #update discriminator on fake samples

        X = np.concatenate((X_real, X_fake), axis=0)
        y = np.concatenate((y_real, y_fake), axis=0)

        X, y = shuffle_inputs(X, y)

        loss, acc = model.train_on_batch(X, y)
        d_losses.append(loss)
        d_accs.append(acc)
        #summarize performance
        print('>Epoch: %d d_loss=%.10f, acc=%.0f%%' % (i+1, loss, acc*100))
    model.trainable = False        
    return d_losses, d_accs

    
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
