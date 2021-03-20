import csv
import glob
import numpy as np
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
                if self.current_idx + self.num_steps + 1>= len(self.data):
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
                if self.current_idx + self.num_steps + 1>= len(self.data):
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
                if self.current_idx + self.num_steps + 1>= len(self.data):
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
            

def process_fxPro(name, ma_days):
    orig = glob.glob(name)[0]
    bars = []
    closes = []
    with open(orig, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i, row in enumerate(csvreader):
            if i > 0 and float(row[5]) > 0.0:           
                open_p = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                closes.append(close)                    
                if i == 1:
                    max_p = open_p
                    min_p = open_p
                
                
                if i > ma_days + 1+10:
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

    out = []
    print(max_p, min_p)
    for l in bars:
        out.append(list(map(lambda x: x, l)))

    return np.array(out)             
    
def process_original(name, ma_days=5):
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
                closes.append(close)
                if i == 1:
                    max_p = open_p
                    min_p = open_p
                
                
                if i > ma_days + 1:
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
                
    out = []
    print(max_p, min_p)
    for l in bars:
        out.append(list(map(lambda x: x, l)))

    return np.array(out)            
