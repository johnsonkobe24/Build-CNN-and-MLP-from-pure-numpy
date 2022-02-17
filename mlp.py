"""
Spring 2022, 10-707
Assignment-1
Problem 4: MLP
TA in charge: Tiancheng Zhao, soyeonmin

IMPORTANT:
    DO NOT change any function signatures

Feb 2022
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime

# When you submit the code for autograder, comment the load cifar 10 dataset command.
# This is only for experiment.
#from load_cifar import trainX, trainy, testX, testy


def random_normal_weight_init(indim, outdim):
    return np.random.normal(0,1,(indim, outdim))

def random_weight_init(indim,outdim):
    b = np.sqrt(6)/np.sqrt(indim+outdim)
    return np.random.uniform(-b,b,(indim, outdim))

def zeros_bias_init(outdim):
    return np.zeros((outdim,1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(10)]for lab in labels],dtype=np.float32)


class Transform:
    """
    This is the base class. You do not need to change anything.

    Read the comments in this class carefully. 
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass



class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x, train=True):
        self.x=x
        return np.maximum(x, 0.0)

    def backward(self, grad_wrt_out):
        check=self.x>0.0
        dldx=grad_wrt_out*check
        return dldx



class LinearMap(Transform):
    """
    Implement this class
    feel free to use random_xxx_init() functions given on top
    """
    def __init__(self, indim, outdim, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        self.alpha = alpha
        self.lr = lr
        self.W = random_weight_init(indim, outdim)
        self.b = zeros_bias_init(outdim)
        self.change1=0
        self.change2=0
        self.indim=indim
        self.outdim=outdim
        self.dldx=np.zeros(self.indim)
        self.dldw=np.zeros((self.indim,self.outdim))
        self.dldb=np.zeros(self.outdim)
    def forward(self, x):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        self.x=x
        self.h= np.matmul(x, self.W) + self.b.T
        return self.h

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, outdim)
        return shape (batch_size, indim)
        Your backward call should Accumulate gradients.
        """
        batch_size, size = self.x.shape
        self.dldx= np.dot(grad_wrt_out,self.W.T)
        self.dldw = self.dldw+np.transpose(np.dot(np.transpose(grad_wrt_out),self.x))
        self.dldb = self.dldb+np.sum(grad_wrt_out, axis=0, keepdims=True)
        return self.dldx
    
    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """

        new_change2 = self.dldb+ self.alpha*self.change2
        self.b = self.b - self.lr * new_change2.T
        new_change1 = self.dldw+self.alpha*self.change1
        self.W=self.W-self.lr * new_change1
        self.change1=new_change1
        self.change2=new_change2
        
    def zerograd(self):
        # reset parameters
        self.dldx=np.zeros(self.indim)
        self.dldw=np.zeros((self.indim,self.outdim))
        self.dldb=np.zeros(self.outdim)
            
    def getW(self):
    # return weights
        return self.W

    def getb(self):
    # return bias
        return self.b

    def loadparams(self, w, b):
    # Used for Autograder. Do not change.
        self.W, self.b = w, b

class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        self.labels=labels
        self.logits=logits
        self.softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        cross_entropy = np.mean(-np.sum(labels*np.log(self.softmax),axis=1), axis=0)
        return cross_entropy

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        dldo=(self.softmax-self.labels) / self.logits.shape[0]
        return dldo
    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        pass



class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, inp, outp, hiddenlayer=100, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.layers=[]
        self.layers.append(LinearMap(inp, hiddenlayer, alpha,lr))
        self.layers.append(ReLU())
        self.layers.append(LinearMap(hiddenlayer, outp, alpha,lr))
    def forward(self, x, train=True):
    # x shape (batch_size, indim)
        self.activated=[]
        self.x=x
        for layer in self.layers:
            self.x= layer.forward(self.x)
            self.activated.append(self.x)
        return self.activated
    
    def backward(self, grad_wrt_out):
        self.dld=grad_wrt_out
        for layer in self.layers[::-1]:
            self.dld=layer.backward(self.dld)
        return self.dld
            
        
    def step(self):
        for layer in self.layers[::-1]:
            if hasattr(layer, 'step'):
                layer.step()

    def zerograd(self):
        for layer in self.layers:
            if hasattr(layer, 'zerograd'):
                layer.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        l=0
        for layer in self.layers: 
            if hasattr(layer, 'loadparams'):
                layer.loadparams(Ws[l],bs[l])
                l=l+1
                

    def getWs(self):
        """
        Return the weights for each layer
        You need to implement this. 
        Return weights for first layer then second and so on...
        """
        self.Ws=[]
        for layer in self.layers:
            if hasattr(layer, 'getW'):
                self.Ws.append(layer.getW())
        return self.Ws
    def getbs(self):
        """
        Return the biases for each layer
        You need to implement this. 
        Return bias for first layer then second and so on...
        """
        self.bs=[]
        for layer in self.layers:
            if hasattr(layer, 'getb'):                
                self.bs.append(layer.getb())
        return self.bs



class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, inp, outp, hiddenlayers=[100,100], alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.layers=[]
        self.layers.append(LinearMap(inp, hiddenlayers[0], alpha,lr))
        self.layers.append(ReLU())
        self.layers.append(LinearMap(hiddenlayers[0], hiddenlayers[1], alpha,lr))
        self.layers.append(ReLU())
        self.layers.append(LinearMap(hiddenlayers[1], outp, alpha,lr))
    def forward(self, x, train=True):
        self.activated = []
        self.x=x
        for layer in self.layers:
            self.x= layer.forward(self.x)
            self.activated.append(self.x)
        return self.activated

    def backward(self, grad_wrt_out):
        self.dld=grad_wrt_out
        for layer in self.layers[::-1]:
            self.dld=layer.backward(self.dld)
        return self.dld

    def step(self):
        for layer in self.layers[::-1]:
            if hasattr(layer, 'step'):
                layer.step()

    def zerograd(self):
        for layer in self.layers:
            if hasattr(layer, 'zerograd'):
                layer.zerograd()

    def loadparams(self, Ws, bs):
        l=0
        for layer in self.layers: 
            if hasattr(layer, 'loadparams'):
                layer.loadparams(Ws[l],bs[l])
                l=l+1

    def getWs(self):
        self.Ws=[]
        for layer in self.layers:
            if hasattr(layer, 'getW'):
                self.Ws.append(layer.getW())
        return self.Ws

    def getbs(self):
        self.bs=[]
        for layer in self.layers:
            if hasattr(layer, 'getb'):                
                self.bs.append(layer.getb().T)
        return self.bs


class Dropout(Transform):
    """
    Implement this class
    """
    def __init__(self, p=0.5):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p=p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing
        """
        self.x=x
        self.keep=1.0-self.p
        if train==True:
            self.mask=np.random.binomial(1 ,self.keep, size=x.shape)
            x=x*self.mask
            self.out=x
        else:
            self.out=self.x *self.p
        return self.out

    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        grad_wrt_out=grad_wrt_out.reshape(self.x.shape)
        self.grad_input=grad_wrt_out * self.mask 
        return self.grad_input



class BatchNorm(Transform):
    """
    Implement this class
    """
    def __init__(self, indim, alpha=0.9, lr=0.001, mm=0.0001):
        Transform.__init__(self)
        """
        You shouldn't need to edit anything in init
        """
        self.alpha = alpha  # parameter for running average of mean and variance
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr
        self.mm = mm  # parameter for updating gamma and beta
        """
        The following attributes will be tested
        """
        self.var = np.ones((1, indim))
        self.mean = np.zeros((1, indim))

        self.gamma = np.ones((1, indim))
        self.beta = np.zeros((1, indim))

        """
        gradient parameters
        """
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        """
        momentum parameters
        """
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)

        """
        inference parameters
        """
        self.running_mean = np.zeros((1, indim))
        self.running_var = np.ones((1, indim))

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        self.batch_size,self.indim=x.shape
        self.x=x
        if train==True:
            self.mean = np.mean(self.x,axis=0)
            self.var = np.var(self.x,axis=0)
            self.sqrt_v = np.sqrt(self.var+self.eps);
            self.norm = (self.x-self.mean )/self.sqrt_v
            self.out=self.norm*self.gamma+self.beta
            self.running_mean = self.running_mean * self.alpha + self.mean * (1-self.alpha)
            self.running_var= self.running_var * self.alpha +  self.var *  (1-self.alpha)
        else:
            self.test_sqrt_v = np.sqrt(self.running_var+ self.eps) ;
            self.test_norm = (self.x - self.running_mean ) / self.test_sqrt_v
            self.out = self.test_norm *  self.gamma  +  self.beta
        return self.out


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        self.dgamma = (grad_wrt_out * self.norm).sum(axis=0)
        self.dbeta = grad_wrt_out.sum(axis=0)
        
        x_mean = self.x - self.mean
        std_inv = 1. / np.sqrt(self.var+self.eps) 
        d_norm = grad_wrt_out * self.gamma
        d_var = np.sum(d_norm * x_mean  ,axis=0)*-0.5*((std_inv)**3)
        d_mean = np.sum(d_norm * -std_inv,axis=0) + d_var * np.mean(-2.* x_mean,axis=0) 
        self.d_x = d_norm * std_inv + d_var * 2 * x_mean/self.batch_size + d_mean/self.batch_size
        
        return self.d_x


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        Make sure your gradient step takes into account momentum.
        Use mm as the momentum parameter.
        """
        self.mbeta_new = self.dbeta+ self.mm*self.mbeta
        self.beta= self.beta - self.lr * self.mbeta_new
        self.mgamma_new = self.dgamma+self.mm*self.mgamma
        self.gamma=self.gamma-self.lr * self.mgamma_new
        self.mgamma=self.mgamma_new
        self.mbeta=self.mbeta_new

    def zerograd(self):
        # reset parameters
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)

    def getgamma(self):
        # return gamma
        return self.gamma

    def getbeta(self):
        # return beta
        return self.beta

    def loadparams(self, gamma, beta):
        # Used for Autograder. Do not change.
        self.gamma, self.beta = gamma, beta

class SingleLayerMLP_BN(Transform):
    """
    Implement this class
    """
    def __init__(self, inp, outp, hiddenlayer=100, alpha_linear=0.9, alpha_BN=0.9, mm=0.01 ,lr=0.001,lr_BN=0.000001):
        Transform.__init__(self)
        self.layers=[]
        self.layers.append(LinearMap(inp, hiddenlayer, alpha_linear,lr))
        self.layers.append(BatchNorm(hiddenlayer,alpha_BN,lr_BN, mm))
        self.layers.append(ReLU())
        self.layers.append(LinearMap(hiddenlayer, outp, alpha_linear,lr))
    def forward(self, x, train=True):
    # x shape (batch_size, indim)
        self.activated=[]
        self.x=x
        self.x= self.layers[0].forward(self.x)
        self.activated.append(self.x)
        self.x= self.layers[1].forward(self.x,train)
        self.activated.append(self.x)
        self.x= self.layers[2].forward(self.x)
        self.activated.append(self.x)
        self.x= self.layers[3].forward(self.x)
        self.activated.append(self.x)
        return self.activated
    
    def backward(self, grad_wrt_out):
        self.dld=grad_wrt_out
        for layer in self.layers[::-1]:
            self.dld=layer.backward(self.dld)
        return self.dld
            
        
    def step(self):
        for layer in self.layers[::-1]:
            if hasattr(layer, 'step'):
                layer.step()

    def zerograd(self):
        for layer in self.layers:
            if hasattr(layer, 'zerograd'):
                layer.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        l=0
        for layer in self.layers: 
            if hasattr(layer, 'loadparams'):
                layer.loadparams(Ws[l],bs[l])
                l=l+1
class SingleLayerMLP_DP(Transform):
    """
    Implement this class
    """
    def __init__(self, inp, outp, hiddenlayer=100, alpha_linear=0.9, p=0.5, lr=0.01):
        Transform.__init__(self)
        self.layers=[]
        self.layers.append(LinearMap(inp, hiddenlayer, alpha_linear,lr))
        self.layers.append(ReLU())
        self.layers.append(Dropout(p))
        self.layers.append(LinearMap(hiddenlayer, outp, alpha_linear,lr))
    def forward(self, x, train=True):
    # x shape (batch_size, indim)
        self.activated=[]
        self.x=x
        self.x= self.layers[0].forward(self.x)
        self.activated.append(self.x)
        self.x= self.layers[1].forward(self.x)
        self.activated.append(self.x)
        self.x= self.layers[2].forward(self.x, train)
        self.activated.append(self.x)
        self.x= self.layers[3].forward(self.x)
        self.activated.append(self.x)
        return self.activated
    
    def backward(self, grad_wrt_out):
        self.dld=grad_wrt_out
        for layer in self.layers[::-1]:
            self.dld=layer.backward(self.dld)
        return self.dld
            
        
    def step(self):
        for layer in self.layers[::-1]:
            if hasattr(layer, 'step'):
                layer.step()

    def zerograd(self):
        for layer in self.layers:
            if hasattr(layer, 'zerograd'):
                layer.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        l=0
        for layer in self.layers: 
            if hasattr(layer, 'loadparams'):
                layer.loadparams(Ws[l],bs[l])
                l=l+1

if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    from load_cifar import trainX, trainy, testX, testy
    trainX = trainX.astype(float)/255.0 #has shape (batch_size x indim)
    testX = testX.astype(float)/255.0  #has shape (batch_size x indim)

    trainLabels = labels2onehot(trainy)
    testLabels = labels2onehot(testy)
    n_epochs=50
    batch_size=128
    train_log=[]
    test_log=[]
    losstrain_log=[]
    losstest_log=[]
    #'''
    net=4
    start = datetime.now()
    #single layer
    if net==1:
        softmax=SoftmaxCrossEntropyLoss()
        network=SingleLayerMLP(inp=trainX.shape[1],outp=10, hiddenlayer=1000, alpha=0.9, lr=0.001)
    #double layer
    if net==2:
        softmax=SoftmaxCrossEntropyLoss()
        network=TwoLayerMLP(inp=trainX.shape[1],outp=10, hiddenlayers=[1000,1000],alpha=0.9, lr=0.001)
    if net==3:
        softmax=SoftmaxCrossEntropyLoss()
        network=SingleLayerMLP_BN(inp=trainX.shape[1],outp=10, hiddenlayer=1000, alpha_linear=0.9,alpha_BN=0.9, mm=0.01 ,lr_BN=0.0001,lr=0.001)
    if net==4:
        softmax=SoftmaxCrossEntropyLoss()
        network=SingleLayerMLP_DP(inp=trainX.shape[1],outp=10, hiddenlayer=1000, alpha_linear=0.9,p=0.5, lr=0.001)
    for epoch in range(n_epochs):
        idxs = np.random.permutation((len(trainLabels)))
        batches=[]
        preds=[]
        lossall=0
        for start_idx in range(0, len(trainX), batch_size):
            if start_idx+batch_size>len(trainX):
                batch_idx = idxs[start_idx: len(trainX)]
                batches.append([trainX[batch_idx], trainLabels[batch_idx]])
            else:
                batch_idx = idxs[start_idx: start_idx + batch_size]
                batches.append([trainX[batch_idx], trainLabels[batch_idx]])
        for batch in batches:
            logits=network.forward(batch[0],train=True)[-1]
            loss=softmax.forward(logits, batch[1])
            dloss=softmax.backward()
            lossall+=loss
            dldx=network.backward(dloss)
            pred=logits.argmax(axis=-1)
            network.step()
            network.zerograd()
            preds=np.append(preds,pred)
        losstrain_log.append(lossall/len(batches))
        logit_test=network.forward(testX,train=False)[-1]
        predict_test=logit_test.argmax(axis=-1)
        loss_test=softmax.forward(logit_test, testLabels)
        losstest_log.append(loss_test)
        train_log.append(sum(preds==trainy[idxs])/len(trainy))
        test_log.append(sum(predict_test==testy)/len(testy))
    
    print(datetime.now() - start)
    fig, host = plt.subplots(figsize=(8,5))
    host.set_ylim(0, 1)
    par1 = host.twinx()
    par1.set_ylim(0, max(losstest_log))
    host.set_ylabel("accuracy")
    par1.set_ylabel("loss")
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    p1, = host.plot(test_log,color=color1, label="test_accuracy")
    p2, = par1.plot(losstest_log, color=color2, label="test_loss")
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')
    plt.show()
    
    fig1, host1 = plt.subplots(figsize=(8,5))
    host1.set_ylim(0, 1)
    par11 = host1.twinx()
    par11.set_ylim(0, max(losstrain_log))
    host1.set_ylabel("accuracy")
    par11.set_ylabel("loss")
    color3 = plt.cm.viridis(.7)
    color4 = plt.cm.viridis(.9)
    p3, = host1.plot(train_log,color=color3, label="train_accuracy")
    p4, = par11.plot(losstrain_log, color=color4, label="train_loss")
    lns1 = [ p3, p4]
    host1.legend(handles=lns1, loc='best')
    plt.show()
    print(test_log[-1], train_log[-1],losstest_log[-1],losstrain_log[-1])

a = np.asarray([test_log, train_log, losstest_log,losstrain_log])
np.savetxt("/Users/johnson/Desktop/MLP4.csv", a, delimiter=",")


