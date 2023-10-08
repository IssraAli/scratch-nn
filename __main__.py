# goal: classify 28x28 handwritten digits

# from scratch
#
# Input layer: 28x28 = 784 nodes -> first hidden layer: 10 nodes -> output layer: 10 nodes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

def relu(x):
    return np.maximum (x, 0)

def softmax(vec):
    return (np.exp(vec))/np.sum(np.exp(vec))

def forward_feed(I, W, b):
    # print ('I: ', I)

    I = I * 1e-4
    U = np.matmul(W[0], I) + b[0]
    # # print('U:', U.shape)

    gamma = relu(U)
    V = np.matmul(W[1], gamma) + b[1]

    # print('V: ', V)

    y_hat = softmax(V)
    # # print('y_hat:', y_hat)

    return I, U, gamma, V, y_hat

def cross_entropy(y_hat, y):
    #minvec = np.asarray([[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).T
    # # print(minvec)
    #y_hat = np.maximum(y_hat, minvec)
    return (-np.matmul(y.T, np.log(y_hat))).item()

def backward_pass(W, b, alpha, I, U, gamma, V, y_hat, y):

    # # print("y: ", y)
    # # print('y_hat', y_hat)
    d1 = (y_hat - y)
    
    # # print ('d1: ', d1.shape)
    # # print ('W[1]: ', W[1].shape)
    # # print ('gamma: ', gamma.shape)

    # # print('U:', U.shape)

    W[1] = W[1] - alpha * np.outer(d1, gamma)
    b[1] = b[1] - alpha * d1

    # # print ('qwerty', np.matmul(W[1], d1).shape)
    # # print ('diagU: ', np.diag(U).shape, U.shape)
    # # print ('U:', U)
    # print (U.T[0].tolist())

    # print (d1.shape, W[1].shape, np.diag(U.T[0].tolist()).shape )

    d2 = np.matmul(d1.T, np.matmul(W[1], np.diag(U.T[0].tolist())))

    
    # # print ('d2: ', d2.shape)
    # # print ('asdf: ', np.outer(d2.T, I).shape)

    W[0] = W[0] - alpha * np.outer(d2.T, I)
    b[0] = b[0] - alpha * d2.T

    return W, b

def one_epoch(training_set, alpha, labels, W, b, max_iter = 10000):

    loss_list = []
    for iter in range(0, max_iter):     
        # print ('\n###########################################\nIteration ', iter)
        I = np.asarray(training_set[iter - 1])
        
        # print (np.shape(I))
        I = np.transpose(I)
        y = np.zeros([10, 1])
        y[labels[iter - 1]] = 1

        I, U, gamma, V, y_hat = forward_feed(I, W, b)
        
        W, b = backward_pass(W, b, alpha, I, U, gamma, V, y_hat, y)

        loss_list.append(cross_entropy(y_hat, y))

    return W, b, np.average(loss_list), np.var(loss_list)

def test(test_set, W, b):
    test_set = test_set.sample(frac = 1).reset_index(drop = True)
    pics = [[test_set.loc[i, 'pixel0':'pixel783'].values] for i in range(0 ,len(test_set))]
    labels = test_set.loc[:, 'label'].values.tolist()   
    
    test_loss = []
    correct_count = 0

    for iter in range (0, len(test_set)):
        I = pics[iter]
        I = np.transpose(I)
        y = np.zeros([10, 1])
        y[labels[iter]] = 1
        
        I, U, gamma, V, y_hat = forward_feed(I, W, b)

        test_loss.append(cross_entropy(y_hat, y))
        #print(np.where(y_hat == max(y_hat))[0][0], np.where(y == 1)[0][0])
        # print(y_hat)
        # print(y)
        if (np.where(y_hat == max(y_hat)) == np.where(y == 1)):
            correct_count += 1

    acc = correct_count/len(test_set)
    # print(I)
    # print(y_hat)
    #print(len(test_set))
    return np.average(test_loss), np.var(test_loss), acc

def learn(training_set, test_set, alpha, epochs = 5000):
    W = [np.random.random_sample(size = [10, 784]) - 0.5, np.random.random_sample(size = [10, 10]) - 0.5] # random init
    b = [np.random.random_sample(size = [10, 1]) - 0.5, np.random.random_sample(size = [10, 1]) - 0.5]
    epochlist = []
    avg_loss_list = []
    var_loss_list = []
    avg_test_loss_list = []
    test_var_list = []
    accuracy_list = []
    for epoch in tqdm(range(0, epochs)):

        training_set = training_set.sample(frac = 1).reset_index(drop = True)
        pics = [[training_set.loc[i, 'pixel0':'pixel783'].values] for i in range(0 ,len(training_set))]
        labels = training_set.loc[:, 'label'].values.tolist()
        # randomize the order of the training set
        W, b, avg_loss, var_loss = one_epoch(pics, alpha, labels, W, b, len(pics))
        
        avg_test_loss, test_var, accuracy = test(test_set, W, b)
        
        avg_loss_list.append(avg_loss)
        var_loss_list.append(var_loss)
        avg_test_loss_list.append(avg_test_loss)
        test_var_list.append(test_var)
        epochlist.append(epoch)
        accuracy_list.append(accuracy)
    return avg_loss_list, epochlist, var_loss_list, avg_test_loss_list, test_var_list, accuracy_list, W, b

if __name__ == "__main__":
    rcParams['font.family'] = "Futura PT"
    rcParams['font.weight'] = "book"
    
    data = pd.read_csv('train.csv')
    data = data.sample(frac = 1).reset_index(drop = True)

    training_set = data[:38400]
    test_set = data[38400:]
    test_set = test_set.reset_index(drop=True)
    print(len(test_set))

    # print (data)

    losslist, epochlist, varlist, testloss, testvar, accuracy, W, b = learn(training_set, test_set, 0.0005, epochs=20)#13) lr = 0.000665

    # Training + testing

    gs = gridspec.GridSpec(1, 2, hspace=0.1)
    fig1 = plt.figure(figsize=(12, 6))

    ax1 = fig1.add_subplot(gs[0,0])
    ax2 = fig1.add_subplot(gs[0, 1])

    ax1.plot(epochlist, losslist, label = 'training loss')
    ax1.plot(epochlist, varlist, label = 'training loss variance')

    ax1.plot(epochlist, testloss, label = 'test loss')
    ax1.plot(epochlist, testvar, label = 'test loss variance')
    ax1.legend()
    ax1.set_title('Average loss during each epoch')
    ax1.set_ylabel('Cross entropy loss')
    ax1.set_xlabel('Epoch')
    ax1.set_xticks(np.arange(0, 20, 5))

    ax2.plot(epochlist, accuracy, label = 'test accuracy')
    ax2.legend()
    ax2.set_title('Test accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_xticks(np.arange(0, 20, 5))

    plt.savefig('loss_accuracy.png', dpi = 300)
    # some examples

    gs2 = gridspec.GridSpec(2, 4, hspace=0.1)   
    fig2 = plt.figure(figsize=(12, 6))

    axlist =[]
    test_set = test_set.sample(frac = 1).reset_index(drop = True)
    pics = [[test_set.loc[i, 'pixel0':'pixel783'].values] for i in range(0 ,len(test_set))]
    labels = test_set.loc[:, 'label'].values.tolist()
    
    pix = pics[:8]
    labelset = labels[:8]

    for i in range(0, 4):
        axlist.append(fig2.add_subplot(gs2[0,i]))
        #axlist[i].plot(picset[i])
        
        picset = np.transpose(pix[i])
        I, U, gamma, V, y_hat = forward_feed(picset, W, b)


        axlist[i].text(x = 0, y = 3, s = ('AI prediction: ' + str(np.where(y_hat == max(y_hat))[0][0])))
        pic = [pix[i][0][j:j+28] for j in range(0, 28*28, 28)]
        plt.imshow(pic, cmap = 'binary')
        axlist[i].set_xticks([])
        axlist[i].set_yticks([])

    for i in range(0, 4):
        axlist.append(fig2.add_subplot(gs2[1,i]))
        picset = np.transpose(pix[i+4])

        I, U, gamma, V, y_hat = forward_feed(picset, W, b)
        axlist[i+4].text(x = 0, y = 3, s = ('AI prediction: ' + str(np.where(y_hat == max(y_hat))[0][0])))
        pic = [pix[i+4][0][j:j+28] for j in range(0, 28*28, 28)]
        plt.imshow(pic, cmap = 'binary')
        axlist[i+4].set_xticks([])
        axlist[i+4].set_yticks([])

    plt.savefig('samples.png', dpi = 300)
    plt.show()



