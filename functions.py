import tensorflow as tf
import keras.backend as K

def my_gMAE_l(y_true, y_pred):
    
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-1]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-1]
    
    lows_real = tf.transpose(realx1)[2]
    lows_pred = tf.transpose(predx1)[2]
        
    subs = tf.math.subtract(lows_pred, lows_real)
    _abs = tf.math.abs(subs)
   
    gMAE_lows = tf.math.reduce_mean(subs)
    
    return gMAE_lows
    
    
def my_gMAE_h(y_true, y_pred):
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-1]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-1]
    
    highs_real = tf.transpose(realx1)[1]
    highs_pred = tf.transpose(predx1)[1]
        
    subs = tf.math.subtract(highs_pred, highs_real)
    _abs = tf.math.abs(subs)
   
    gMAE_highs = tf.math.reduce_mean(subs)
    
    return gMAE_highs
    
def my_gMSE(y_true, y_pred):

    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-1]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-1]

    subs = tf.math.subtract(predx1, realx1)

    squares = tf.math.multiply(subs, subs)

    gMSE = tf.math.reduce_mean(squares)
    return gMSE
    
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
    eqs = tf.cast(eqs, tf.float32)
    
    return tf.math.reduce_mean(eqs)

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
    Xreal = Xreal + K.epsilon()
    
    log_real = tf.math.log(Xreal)
    
    fin_real = tf.math.reduce_mean(log_real)
    
    
    #get indexes of fake y
    eqs = tf.math.equal(y_trues, 0)
    idx = tf.where(eqs)
    
    #predictions for when data was fake
    Xfake = tf.gather(y_preds, idx)

    sub_fake = tf.math.subtract(1.0, Xfake)
    sub_fake = sub_fake + K.epsilon()

    log_fake = tf.math.log(sub_fake)
    
    fin_fake = tf.math.reduce_mean(log_real)

    return -fin_real - fin_fake
    
def my_aloss(y_true,y_pred):
    
    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-2]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-2]
    
    subs = tf.math.subtract(predx1, realx1)
    squares = tf.math.multiply(subs, subs)
    
    #remove last column with Mean average
    residual = tf.transpose(squares)[:-1]

    #compute mean
    gMSE = tf.math.reduce_mean(residual)

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
    sub_fake_cor = tf.math.add(sub_fake, K.epsilon())
    log_fake = tf.math.log(sub_fake)
    
    gloss = tf.math.reduce_mean(log_fake)   
    

    #maybe gloss hyperparameter should be negative!
    h1 = 1
    h2 = 1
    return h1*gMSE + h2*gloss
