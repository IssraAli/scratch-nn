# goal: calssify 28x28 handwritten digits

# from scratch
#
# Input layer: 28x28 = 784 nodes -> first hidden layer: 10 nodes -> output layer: 10 nodes
#
# 

import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd

def relu(x):
    return cp.maximum (x, 0)

def softmax(vec):
    
    return (cp.exp(vec))/cp.sum(cp.exp(vec))

def forward_feed(I, W, b):
    print ('I: ', I)
    I = I * 1e-3
    U = cp.matmul(W[0], I) + b[0]
    # print('U:', U.shape)

    gamma = relu(U)
    V = cp.matmul(W[1], gamma) + b[1]

    print('V: ', V)

    y_hat = softmax(V)
    # print('y_hat:', y_hat)

    if cp.nan in (I) or cp.nan in (U) or cp.nan in (gamma) or cp.nan in (V) or cp.nan in (y_hat):
        print ('!') 

    return I, U, gamma, V, y_hat

def cross_entropy(y_hat, y):
    minvec = cp.asarray([[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).T
    # print(minvec)
    y_hat = cp.maximum(y_hat, minvec)
    return (-cp.matmul(y.T, cp.log(y_hat))).item()

def backward_pass(W, b, alpha, I, U, gamma, V, y_hat, y):

    # print("y: ", y)
    # print('y_hat', y_hat)
    d1 = (y_hat - y)
    
    # print ('d1: ', d1.shape)
    # print ('W[1]: ', W[1].shape)
    # print ('gamma: ', gamma.shape)

    # print('U:', U.shape)

    W[1] = W[1] - alpha * cp.outer(d1, gamma)
    b[1] = b[1] - alpha * d1

    # print ('qwerty', cp.matmul(W[1], d1).shape)
    # print ('diagU: ', cp.diag(U).shape, U.shape)
    # print ('U:', U)
    print (U.T[0].tolist())

    print (d1.shape, W[1].shape, cp.diag(U.T[0].tolist()).shape )

    d2 = cp.matmul(d1.T, cp.matmul(W[1], cp.diag(U.T[0].tolist())))

    
    # print ('d2: ', d2.shape)
    # print ('asdf: ', cp.outer(d2.T, I).shape)

    W[0] = W[0] - alpha * cp.outer(d2.T, I)
    b[0] = b[0] - alpha * d2.T

    return W, b

def learn(training_set, alpha, labels, max_iter = 10000):
    iter = 0
    
    W = [cp.random.random_sample(size = [10, 784]) - 0.5, cp.random.random_sample(size = [10, 10]) - 0.5] # random init
    b = [cp.random.random_sample(size = [10, 1]) - 0.5, cp.random.random_sample(size = [10, 1]) - 0.5]

    iter = 1

    iterlist = []
    losslist = []

    while 0 < iter < max_iter:
        
        print ('\n###########################################\nIteration ', iter)
        I = training_set[iter - 1]
        print(I, I.shape)
        I = cp.transpose(I)
        y = cp.zeros([10, 1])
        y[labels[iter - 1]] = 1

        I, U, gamma, V, y_hat = forward_feed(I, W, b)
        
        losslist.append(cross_entropy(y_hat, y))
        iterlist.append(iter)

        print('old Wmax:', cp.amax(W[0]))
        
        W, b = backward_pass(W, b, alpha, I, U, gamma, V, y_hat, y)

        print('new Wmax:', cp.amax(W[0]))

        iter += 1
    
    return losslist, iterlist


if __name__ == "__main__":
    training_set = pd.read_csv('train.csv')
    pics = [[training_set.loc[i, 'pixel0':'pixel783'].values] for i in range(0 ,len(training_set))]
    labels = training_set.loc[:, 'label'].values.tolist()

    # print (pics[:2])
    # print (cp.shape(pics[1]))
    # print (type(pics[1]))
    

    print('asdf:', softmax(cp.asarray([1, 2, 3, 4])))

    data = pd.DataFrame({'labels' : labels, 'pics' : pics})
    print (data)

    losslist, iterlist = learn(pics, 0.001, labels, max_iter=30000)

    print (losslist)

    plt.plot(iterlist, losslist)

    plt.show()



