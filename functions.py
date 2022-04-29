import tensorflow as tf
import keras.backend as K

def my_mDir(y_true, y_pred):
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-1]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-1]
    open_preds = tf.transpose(predx1)[0]
    close_preds = tf.transpose(predx1)[3]
    sub_preds = tf.math.subtract(open_preds, close_preds)

    open_reals = tf.transpose(realx1)[0]
    close_reals = tf.transpose(realx1)[3]    
    sub_reals = tf.math.subtract(open_reals, close_reals)
    mults = tf.math.multiply(sub_preds, sub_reals)
    correct_direction = tf.math.greater(mults, 0)
    tonum = tf.cast(correct_direction, tf.float32)

    market_direction = tf.math.reduce_mean(tonum)
    return market_direction
    
def my_gMAE(y_true, y_pred):
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-1]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-1]
    subs = tf.math.subtract(predx1, realx1)
    _abs = tf.math.abs(subs)
    gMAE = tf.math.reduce_mean(_abs)
    return gMAE
    
   

def my_gMSE_o_c(y_true, y_pred):
    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-1]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-1]

    opens_real = tf.transpose(realx1)[0]
    opens_pred = tf.transpose(predx1)[0]

    subs_opens = tf.math.subtract(opens_real, opens_pred)

    squares_opens = tf.math.multiply(subs_opens, subs_opens)

    gMSE_opens = tf.math.reduce_mean(squares_opens)
    
    closes_real = tf.transpose(realx1)[3]
    closes_pred = tf.transpose(predx1)[3]

    subs_closes = tf.math.subtract(closes_real, closes_pred)

    squares_closes = tf.math.multiply(subs_closes, subs_closes)

    gMSE_closes = tf.math.reduce_mean(squares_closes)
    
    return gMSE_opens + gMSE_closes

#WARNING MSE should be True - Prediction, not the other way around
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
    #tf.print(11111111111111)
    #tf.print(y_preds)
#    tf.print(y_trues, summarize=-1)
#    tf.print(y_preds, summarize=-1)    
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
    
    fin_fake = tf.math.reduce_mean(log_fake)

    return -fin_real - fin_fake

def oc_aloss(y_true,y_pred):
    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-2]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-2]
    
    subs = tf.math.subtract(predx1, realx1)
    squares = tf.math.multiply(subs, subs)
    
    transposed = tf.transpose(squares)
    
    
    opens = tf.math.reduce_mean(transposed[0])
    highs = tf.math.reduce_mean(transposed[1])
    lows = tf.math.reduce_mean(transposed[2])
    closes = tf.math.reduce_mean(transposed[3])    
    
#    future_closes = tf.math.reduce_mean(transposed[3])
    
#    gMSE = (closes + highs + lows + opens) / 4
    gMSE = (closes+ opens) / 2
    
    predx12 = tf.transpose(y_pred, perm=[1,0,2])[-2]
    open_preds = tf.transpose(predx12)[0]
    close_preds = tf.transpose(predx12)[3]
    sub_preds = tf.math.subtract(open_preds, close_preds)
    
    realx12 = tf.transpose(y_true, perm=[1,0,2])[-2]
    open_reals = tf.transpose(realx12)[0]
    close_reals = tf.transpose(realx12)[3]    
    sub_reals = tf.math.subtract(open_reals, close_reals)
    
    #chceme ak je sub_preds kladne, aby aj sub_reals bolo kladne
    #chcema ke je sub_preds zaporne, aby aj sub_reals bolo zaporne
    #Vynasobenim dostaneme kladne cislo ak maju rovnake znamienka,
    #zaporne cislo ak maju rozne znamienka
    mults = tf.math.multiply(sub_preds, sub_reals)
    correct_direction = tf.math.greater(mults, 0)  
#    tf.print(mults, summarize=-1)
#    mults_cor = tf.math.add(mults, K.epsilon())
#    log_mults = tf.math.log(mults_cor)
#    tf.print(log_mults, summarize=-1)

    tonum = tf.cast(correct_direction, tf.float32)
    market_loss = tf.math.reduce_mean(tonum)
#    tf.print(market_loss, summarize=-1)    
    
    
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
    log_fake = tf.math.log(sub_fake_cor)
    
    gloss = tf.math.reduce_mean(log_fake)   

    #gMSE is around -1 to 1
    #gloss is -17 to 0
    #we should scale so its of the same importance

    h1 = 10
    h2 = 1
    h3 = 10
    return h1*gMSE + h2*gloss - h3*market_loss

    
def my_aloss(y_true,y_pred):
    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-2]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-2]
#    tf.print(predx1)
#    tf.print(realx1)
    subs = tf.math.subtract(predx1, realx1)
#    tf.print("AAAAAAAA")
#    tf.print(subs)
    _abs = tf.math.abs(subs)
    squares = tf.math.multiply(subs, subs)
#    tf.print('BBBBBB')
#    tf.print(squares)
    #remove last column with Mean average
#    residual = tf.transpose(squares)[:-2]

    #compute mean
#    gMSE = tf.math.reduce_mean(residual)
#    gMSE = tf.math.reduce_mean(squares)
    gMAE = tf.math.reduce_mean(_abs)
   
    #---------------gloss-------------------------------------------------#
    #We want to penalise the generator if it doesnt fool the discriminator
    #In case the discriminator outputs 0 (or number close to 0), it thinks the data is faked
    #We compute log(1-prediction)
    #If it predicts 0 (i.e. 'fake'), we compute log(1-0) = log(1) = 0
    #If it predicts 0.9(i.e 'real'), we compute log(1-0.9) = log(0.1) = -2.3
    #This way, the more we fool the discriminator the more we decrease loss
    #--------------gloss--------------------------------------------------#
    y_preds = tf.transpose(y_pred, perm=[1,0,2])[-1]
    Xfake = tf.transpose(y_preds)[0]    
    sub_fake = tf.math.subtract(1.0, Xfake)
    sub_fake_cor = tf.math.add(sub_fake, K.epsilon())
    log_fake = tf.math.log(sub_fake_cor)
    gloss = tf.math.reduce_mean(log_fake)   

    #---------------market_direction---------------------------------------------------------#
    #We want to penalise the generator if it predicts wrong market direction.
    #By market direction, we mean simply If the Close of the day is above or
    #under Open
    #Therefore, if real Open - Close is positive, we want predicted Open - Close positive
    #If we get the substitutions and multiply them,i.e.
    #(OpenReal - CloseReal) * (OpenPred - ClosePred)
    #We get an array where positive numbers mean correct direction predicted and negative numbers
    #incorrect direction [-1.341, 0.314155, -2.3415, -3.1345]
    #Then we can turn the numbers to boolean with math.greater(0) to have Zeros if negative and
    #Ones if positive.[0,1,0,0]
    #Then we cast that to float. [0.0, 1.0, 0.0, 0.0]
    #And compute mean. Problem is when the predictions are good, we're getting higher number,
    #so we'd just take negative of that
    #---------------market_direction---------------------------------------------------------#
    open_preds = tf.transpose(predx1)[0]
    close_preds = tf.transpose(predx1)[3]
    sub_preds = tf.math.subtract(open_preds, close_preds)

    open_reals = tf.transpose(realx1)[0]
    close_reals = tf.transpose(realx1)[3]    
    sub_reals = tf.math.subtract(open_reals, close_reals)
    mults = tf.math.multiply(sub_preds, sub_reals)
    correct_direction = tf.math.greater(mults, 0)
    tonum = tf.cast(correct_direction, tf.float32)

    market_direction = tf.math.reduce_mean(tonum)
    market_direction2 = tf.math.multiply(market_direction, market_direction)

    h1 = 10
    h2 = 1
    h3 = 100
    return h1*gMAE + h2*gloss# - h3*market_direction2

def weighted_aloss(y_true,y_pred):
    
    #gMSE
    predx1 = tf.transpose(y_pred, perm=[1,0,2])[-2]
    realx1 = tf.transpose(y_true, perm=[1,0,2])[-2]
    
    subs = tf.math.subtract(predx1, realx1)
    
    t_subs = tf.transpose(subs)
    
    #subs[2] -->lows
    #we want to buy on low, so the actual price can't go much lower than predicted low
    #if true low is lower than predicted, predx1 > realx1, subs will be positive
    #if subs in low column is positive, we'll square it

    
    positives = tf.cast(tf.math.greater(t_subs[2], 0), tf.float32)
    #tf.print(positives)
    positives = positives * t_subs[2]
    #tf.print(positives, summarize=-1)
    #tf.print(t_subs[2] + positives)
    lows = t_subs[2] + positives
#    lows = t_subs[2]
    lows = tf.math.multiply(lows, lows)
    
    #subs[1] -->highs
    negatives = tf.cast(tf.math.less(t_subs[1], 0), tf.float32)
#    tf.print(positives)
    negatives = negatives * t_subs[1]
#    tf.print(positives, summarize=-1)
#    tf.print(t_subs[1] + positives)
#    t_subs[1] = t_subs[1] + positives
#    tf.print(t_subs)
    highs = t_subs[1] + negatives

#    highs = t_subs[1]
    highs = tf.math.multiply(highs, highs)
    
    closes = t_subs[3]
    closes = tf.math.multiply(closes, closes)

    closes = tf.math.reduce_mean(closes)    
    highs = tf.math.reduce_mean(highs)
    lows = tf.math.reduce_mean(lows)
    
    gMSE = (closes + highs + lows) / 3
    

    
#    squares = tf.math.multiply(subs, subs)
    
    #remove last column with Mean average
#    residual = tf.transpose(squares)[:-2]

#    gMSE = tf.math.reduce_mean(residual)

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
    log_fake = tf.math.log(sub_fake_cor)
    
    gloss = tf.math.reduce_mean(log_fake)   

    #gMSE is around -1 to 1
    #gloss is -17 to 0
    #we should scale so its of the same importance

    h1 = 1
    h2 = 1
    return h1*gMSE + h2*gloss
