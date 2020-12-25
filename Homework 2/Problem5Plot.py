import numpy as np
import matplotlib.pyplot as plt
import Problem5

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each part
    data = np.load('data.npy')
    x_data, y_data = np.split(data, 2, 1)
    
    n, p = 0, 0
    
    if len(x_data.shape) > 1:
        n = x_data.shape[1]
    else:
        n = 1
        
    if len(y_data) > 1:
        p = y_data.shape[1]
    else:
        p = 1
    
    m = x_data.shape[0]
    weights = np.random.rand(n, p)
    
    w1, hist1 = Problem5.bgd_l2(x_data, y_data, weights, 0.05, 0.10, 0.001, 50)
    
    plt.clf()
    plt.plot(hist1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("GD")

