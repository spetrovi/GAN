import tensorflow as tf
import keras.backend as K

def my_dacc(y_true, y_pred):
    #from every element of batch, elem, we want to take elem[-1][0]
    #for this we need to do some transpositions, to select correct data
    y_trues = tf.transpose(y_true, perm=[1,0,2])[-1]
    y_trues = tf.transpose(y_trues)[0]
    y_trues = tf.math.greater(y_trues, 0.5)  
    
    y_preds = tf.transpose(y_pred, perm=[1,0,2])[-1]
    y_preds = tf.transpose(y_preds)[0]    
    y_preds = tf.math.greater(y_preds, 0.5)  
    
#    tf.print(y_trues, summarize=-1)
#    tf.print(y_preds, summarize=-1)

    eqs = tf.math.equal(y_trues, y_preds)
    eq_sum = tf.reduce_sum(tf.cast(eqs, tf.float32))
    
    return eq_sum/y_trues.shape[0]

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
    gMSE = tf.math.multiply(xsum, 1/(y_true.shape[2]*y_true.shape[0]))

    #---------------gloss-------------------------------------------------#
    #We want to penalise the generator if it doesnt fool the discriminator
    #In case the discriminator outputs 0 (or number close to 0), it thinks the data is faked
    #if it think it's fake, we will compute logarithm (1), which is 0
    #if it thinks its real(0.9), we will compute logarithm (0.1), which is -2.3
    #Therefore, the more we fool the discriminator, the more we decrease the loss
    #--------------gloss--------------------------------------------------#
    y_preds = tf.transpose(y_pred, perm=[1,0,2])[-1]

    Xfake = tf.transpose(y_preds)[0]    
    sub_fake = tf.math.subtract(1.0, Xfake)
    log_fake = tf.math.log(sub_fake)
    sum_fake = tf.math.reduce_sum(log_fake)
    gloss = tf.math.multiply(sum_fake, 1/y_true.shape[0])
    

    #maybe gloss hyperparameter should be negative!
    h1 = 1
    h2 = 1
    return h1*gMSE + h2*gloss


def my_mse(y_true,y_pred):
    y_true = tf.transpose(y_true)[-1]
    y_pred = tf.transpose(y_pred)[-1]
    return K.mean(K.square(y_pred-y_true))
