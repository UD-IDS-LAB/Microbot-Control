import numpy as np

class mean_val:
    def __init__(self):
        self.k = 0
        self.val = 0
        self.mean = 0
        
    def append(self,x):
        self.k += 1
        self.val += x
        self.mean = self.val/self.k
        
    def get(self):
        return self.mean

class logger:
    """ Using logger will help you to save all the data at one place and pass is as a single variable """
    def __init__(self):
        self.log = dict()
        
    def add_log(self,name):
        self.log[name] = []
        
    def add_item(self,name,x):
        self.log[name].append(x)
        
    def get_log(self,name):
        return self.log[name]
    
    def get_keys(self):
        return self.log.keys()
    
    def get_current(self,name):
        return self.log[name][-1]
    
    def get_latest_log(self, name, latest = 100):
        return self.log[name][(len(self.log[name])-latest):]

def smooth(x,window_len=11,window='hanning'):
    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y