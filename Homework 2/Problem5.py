import math
import numpy as np

# Batch Gradient Descent
def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []
    new_w = [np.copy(w)]
    n = data.shape[0]
    
    for i in range(num_iter):
        temp, other = 0, 0
        
        for k in range(0, n):
            mult = np.multiply(data[k], new_w)
            
            if (y >= (mult + delta)).all():
                temp = temp + (y[k] - mult - delta)**2
                other = other + 2 * (y[k] - mult - delta).dot(data)
            elif (abs(y - mult) < delta).all():
                temp += 0
                other += 0
            else:
                temp = temp + (y[k] - mult + delta)**2
                other = other + 2 * np.dot((y[k]- mult + delta), data[k])
            
        temp *= 1/n
        other *= 1/n
        
        history_fw.append(temp + lam * np.dot(new_w, (np.transpose(w)))[0, 0])
        new_w = new_w + (eta * (other + (lam * 2 * np.sum(w))))
        
    return new_w, history_fw

# Stochastic Gradient Descent
def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    history_fw = []
    new_w = np.copy(w)
    n = data.shape[0]
    s = i
    
    for j in range(num_iter):
        if i == -1:
            s = np.random.randint(0, n)    
        
        other, temp = 0, 0
        
        mult = np.multiply(data[s], new_w)
        
        if y >= (mult + delta):
            temp += (y - mult - delta)**2
            other += 2 * (y - mult - delta).dot(data)
        elif abs(y - mult) < delta:
            temp += 0
            other += 0
        else:
            temp += (y - mult + delta)**2
            other += 2 * (y - mult + delta).dot(data)
        
        other *= 1/n
    
    history_fw.append(temp + lam * new_w.dot(np.transpose(w)[0, 0]))
    new_w += (eta/math.sqrt(j)) * (other + lam * 2 * np.sum(new_w))
        
    return new_w, history_fw