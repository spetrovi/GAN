import csv
import glob
import numpy as np
import random

class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, num_params, name, skip_step=1):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_params = num_params
        self.name = name
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate_for_dmodel_real(self):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.ones((self.batch_size, self.num_steps + 1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps + 1>= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]

                x[i, :] = x_days.copy()                    
                y[i, :] = x_days.copy()
                
                self.current_idx += self.skip_step


            #normalise
            mean = np.mean(x)
            std = np.std(x)
            x = (x - mean) / std
            _min = np.abs(np.min(x))
            x = x + _min

            #add row of zeroes to every sequence
            y = list(x.copy())
            for i in range(self.batch_size):                
                y[i] = np.vstack((y[i], np.ones(self.num_params)))

            #move cursor to different point in data                        
            self.current_idx = random.randint(0, len(self.data)-self.num_steps+1)       
            yield (x, y)

    #Generate for training d model
    def generate_for_dmodel_fake(self, g_model):
        x_for_g = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps + 1>= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                
                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                y_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]                

                x_for_g[i, :] = x_days.copy()                    
                y[i, :] = y_days.copy()
                
                self.current_idx += self.skip_step
                
            #normalise
            mean = np.mean(y)
            std = np.std(y)
            
            y = (y - mean) / std
            _min = np.abs(np.min(y))
            y = y + _min
            
            x_for_g = (x_for_g - mean) / std
            x_for_g = x_for_g + _min
            
            x = g_model.predict(x_for_g)

            #add row of zeroes
            y = list(y)
            for i in range(self.batch_size):                
                y[i] = np.vstack((y[i], np.zeros(self.num_params)))

            #move cursor to different point in data
            self.current_idx = random.randint(0, len(self.data)-self.num_steps+1)
            yield (x, y)
            
    def generate_for_amodel(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.ones((self.batch_size, self.num_steps+1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps + 1>= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                y_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]

                x[i, :] = x_days.copy()
                y[i, :] = y_days.copy()
                
                self.current_idx += self.skip_step

            #normalise
            mean = np.mean(y)
            std = np.std(y)
            
            y = (y - mean) / std
            _min = np.abs(np.min(y))
            y = y + _min
            
            x = (x - mean) / std
            x = x + _min            
            

            
            #add row of zeroes
            y = list(y)
            for i in range(self.batch_size):                
                y[i] = np.vstack((y[i], np.zeros(self.num_params)))

            y = np.array(y)
            x = np.array(x)
            #move cursor to different point in data
            self.current_idx = random.randint(0, len(self.data)-self.num_steps+1)
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
                
                x[i, :] = x_days.copy()
                y[i, :] = y_days.copy()

                self.current_idx += self.skip_step
                
            #normalise
            mean = np.mean(y)
            std = np.std(y)
            
            y = (y - mean) / std
            _min = np.abs(np.min(y))
            y = y + _min
            
            x = (x - mean) / std
            x = x + _min            
            
            self.current_idx = random.randint(0, len(self.data)-self.num_steps+1)
            yield (x, y)
            
  


def process_fxPro(name, ma_days):
    print(name)
    orig = glob.glob(name)[0] 
    bars = []
    volumes = []
    with open(orig, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i, row in enumerate(csvreader):
            if i > 0 and float(row[5]) > 0.0:           
                open_p = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                bars.append([open_p, high, low, close])
                volumes.append(float(row[5]))

    #compute indicators
    out = []
    for i in range(ma_days, len(bars)):
        row = bars[i].copy()


        # dataset[0:5] > indexes 0..4, next value is at dataset[5]
        #mean average
        ma = np.mean(bars[i-ma_days:i + 1])
        row.append(ma)

        t_days = np.transpose(bars[i-ma_days:i + 1])
        
        #mean high
        h_mean = np.mean(t_days[1])
        row.append(h_mean)

        #max high
        h_max = np.max(t_days[1])
        row.append(h_max)
        
        #mean low
        l_mean = np.mean(t_days[2])
        row.append(l_mean)
        
        #min low
        l_min = np.min(t_days[2])
        row.append(l_min)
        #append next days' open
#        row.append(bars[i+1][0])
        
        #volume
#        row.append(volumes[i])

        out.append(row)

    return name,  np.array(out)
    
def process_fxPro_hourly(name, ma_days):
    print(name)
    orig = glob.glob(name)[0] 
    bars = []
    volumes = []
    with open(orig, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i, row in enumerate(csvreader):
            if i > 0 and float(row[6]) > 0.0:           
                open_p = float(row[2])
                high = float(row[3])
                low = float(row[4])
                close = float(row[5])
                bars.append([open_p, high, low, close])
                volumes.append(float(row[6]))

    #compute indicators
    out = []
    for i in range(ma_days, len(bars)):
        row = bars[i].copy()
        

        # dataset[0:5] > indexes 0..4, next value is at dataset[5]
        #mean average
        ma = np.mean(bars[i-ma_days:i + 1])
        row.append(ma)

        t_days = np.transpose(bars[i-ma_days:i + 1])
        
        #mean high
        h_mean = np.mean(t_days[1])
        row.append(h_mean)

        #max high
        h_max = np.max(t_days[1])
        row.append(h_max)
        
        #mean low
        l_mean = np.mean(t_days[2])
        row.append(l_mean)
        
        #min low
        l_min = np.min(t_days[2])
        row.append(l_min)
        
        #append next days' open
#        row.append(bars[i+1][0])
        
        #volume
#        row.append(volumes[i])

        out.append(row)

    return name,  np.array(out)
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

    return name, np.array(out)            
