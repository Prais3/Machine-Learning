import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    
    init_centers = []
    lst = [np.random.randint(0, X.shape[0])]
    init_centers.append(X[lst[0]])
    
    for center in range(1, k):
        dists = cdist(X, np.array([init_centers[-1]]))
        updated_center = np.where((dists / dists.max()) > np.random.rand())[0]
        
        if np.isin(updated_center, lst).all():
            init_centers.append(X[updated_center[0]])
        else:
            for n in updated_center:
                if n not in lst:
                    lst.append(n)
                    init_centers.append(X[n])
                    break
                
    return np.array(init_centers)

        
def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    curr = k_init(X, k)
    final_centers = []
    
    for i in range(max_iter):
        assign_data = assign_data2clusters(X, curr)
        new = np.zeros(curr.shape)
        
        for c in range(curr.shape[0]):
            count_items = 0
            val = np.zeros(X.shape[1])
            
            for j in range(X.shape[0]):
                if assign_data[j][c] == 1:
                    val += X[j]
                    count_items += 1
                
                if count_items > 0:
                    new[c] = val / count_items
                else:
                    new[c] = curr[c]
            
        final_centers = new
        
        if (curr == final_centers).all():
            break
        else:
            curr = final_centers
        
        return final_centers


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    if C.shape[0] == 1:
        C = np.array([C[-1]])
        
    dists = sp.spatial.distance.cdist(X, C)
    
    data_map = np.zeros(dists.shape)
    data_map[np.arange(dists.shape[0]), np.argmin(dists, axis=1)] = 1
    
    return data_map


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    
    row_x = X.shape[0]
    row_c = C.shape[0]
    
    input_err = np.zeros(row_x)
    loc = assign_data2clusters(X, C)
    
    for c in range(row_c):
        for i in range(row_x):
            if loc[i][c] == 1:
                diff = X[i] - C[c]
                input_err[i] = np.linalg.norm(diff)
    
    accuracy = sum(input_err) / row_x
    return accuracy


# get_data function to load the data from the iris.data file
def get_data():
    data = np.genfromtxt('iris.data', delimiter=',')
    data = np.delete(data, 4, axis=1)
    
    sepal = data[:, 0] / data[:, 1]
    petal = data[:, 2] / data[:, 3]
    
    return np.transpose(np.array([sepal, petal]))

# Call get_data above to get the data
iris_data = get_data()

cluster_acc = []
res = []
curr = []

# for loop for the number of clusters from 1 to 5
for k in range(1, 6):
    center_updated = k_means_pp(iris_data, k, 50)
    cluster_acc.append(compute_objective(iris_data, center_updated))
    
# Simple for loop to record over by running the algorithm 50 times
for i in range(1, 51):
    curr = k_means_pp(iris_data, 3, i)
    res.append(compute_objective(iris_data, curr))
  
labels = np.argmax(assign_data2clusters(iris_data, curr), axis = 1)    

# Plot to observe the clusters
x1 = iris_data[:, 0]
y1 = iris_data[:, 1]
plt.scatter(x = x1, y = y1, c = labels)
plt.xlabel('Sepal')
plt.ylabel('Petal')
plt.show()

# Plot to see total clusters based on new center
plt.plot(np.arange(1, 6), cluster_acc)
plt.ylabel('objective')
plt.xlabel('Total Clusters')
plt.show()

# Objective updated based on the number of iterations plot
plt.plot(np.arange(1, 51), res)
plt.ylabel('result')
plt.xlabel('Number of iterations')
plt.show()