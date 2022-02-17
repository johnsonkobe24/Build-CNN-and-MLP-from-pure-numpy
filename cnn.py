"""
Spring 2022, 10-707
Assignment-1
Problem 5: CNN
TA in charge: Tiancheng Zhao, soyeonmin

IMPORTANT:
    DO NOT change any function signatures

    Some modules in Problem 4 like ReLU and LinearLayer are similar to Problem1
    but not exactly same. Read their commented instructions carefully.

Feb 2022
"""
#%%
import numpy as np
#import copy
# When you submit the code for autograder, comment the load cifar 10 dataset command.
# This is only for experiment.
from load_cifar import trainX, trainy, testX, testy
from datetime import datetime
def labels2onehot(labels):
    return np.array([[i==lab for i in range(10)]for lab in labels],dtype=np.float32)
"""
def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    '''
    
    X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    N, C, H, W = X.shape
    height_wave = (H + 2 * padding - k_height) //stride + 1
    width_wave = (W + 2 * padding - k_width) // stride + 1
    ix = np.tile(np.repeat(np.arange(k_height), k_width),C)
    iy = stride * np.repeat(np.arange(height_wave), width_wave)
    jx = np.tile(np.arange(k_width), k_height * C)
    jy = stride * np.tile(np.arange(width_wave), height_wave)
    i = (ix.reshape(-1, 1) + iy.reshape(1, -1)).astype(int)
    j = (jx.reshape(-1, 1) + jy.reshape(1, -1)).astype(int)
    k = (np.repeat(np.arange(C), k_height * k_width).reshape(-1, 1)).astype(int)
    X_col = X_padded[:, k, i, j]
    C = X.shape[1]
    X_col = X_col.transpose(1, 2, 0).reshape(k_height * k_width * C, -1)
    return X_col
    



def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    '''
    N, C, H, W = X_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=grad_X_col.dtype)
    height_wave = (H + 2 * padding - k_height) //stride + 1
    width_wave = (W + 2 * padding - k_width) // stride + 1
    ix = np.tile(np.repeat(np.arange(k_height), k_width),C)
    iy = stride * np.repeat(np.arange(height_wave), width_wave)
    jx = np.tile(np.arange(k_width), k_height * C)
    jy = stride * np.tile(np.arange(width_wave), height_wave)
    i = (ix.reshape(-1, 1) + iy.reshape(1, -1)).astype(int)
    j = (jx.reshape(-1, 1) + jy.reshape(1, -1)).astype(int)
    k = (np.repeat(np.arange(C), k_height * k_width).reshape(-1, 1)).astype(int)
    grad_X_col_reshaped = grad_X_col.reshape(C * k_height * k_width, -1, N)
    grad_X_col_reshaped = grad_X_col_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), grad_X_col_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

"""
def im2col(input_data, filter_h, filter_w, pad=1 ,stride=1):

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def im2col_bw(col, input_shape, filter_h, filter_w, pad=1, stride=1):

    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

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
        Unlike Problem 1 MLP, here we no longer accumulate the gradient values,
        we assign new gradients directly. This means we should call update()
        every time we do forward and backward, which is fine. Consequently, in
        Problem 2 zerograd() is not needed any more.
        Compute and save the gradients wrt the parameters for update()
        Read comments in each class to see what to return.
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def forward(self, x, train=True):
        """
        returns ReLU(x)
        """
        self.x=x
        return np.maximum(x, 0.0)

    def backward(self, dLoss_dout):
        """
        dLoss_dout is the gradients wrt the output of ReLU
        returns gradients wrt the input to ReLU
        """
        check=self.x>0.0
        dldx=dLoss_dout*check
        return dldx

class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        self.x=x
        reshapped=x.reshape(x.shape[0],-1)
        return reshapped

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        dreshapped=dloss.reshape(self.x.shape)
        return dreshapped


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.C, self.H, self.Width = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        b = np.sqrt(6) / np.sqrt((self.C + self.num_filters) * self.k_height * self.k_width)
        self.W = np.random.uniform(-b, b, (self.num_filters, self.C, self.k_height, self.k_width))
        self.b = np.zeros((self.num_filters, 1))
        self.dldw_momentum=0
        self.dldb_momentum=0
        #self.dldw=np.zeros((self.num_filters,self.C, self.k_height, self.k_width))
        #self.dldx=np.zeros((self.C, self.num_filters, self.H, self.Width))
        #self.dldb=np.zeros(self.b.shape)
        

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here
        """
        self.inputs=inputs
        self.pad=pad
        self.stride=stride
        self.batch_size=inputs.shape[0]
        self.X_col=im2col(inputs, self.k_height, self.k_width, pad, stride)
        self.height_wave = (inputs.shape[2] + 2 * pad - self.k_height) // stride + 1
        self.width_wave = (inputs.shape[3] + 2 * pad- self.k_width) // stride + 1
        self.W_col=self.W.reshape(self.num_filters,-1).T
        Conv_Output = (np.matmul( self.X_col, self.W_col) + self.b.T).reshape(self.batch_size,self.height_wave, self.width_wave,-1).transpose(0,3,1,2)
        return Conv_Output
        
        
        
    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        fdloss=dloss.transpose(0,2,3,1).reshape(-1, self.num_filters) #NHW*F KKC*NHW KKC*F F*CKK 
        self.dldw=(self.X_col.T @ fdloss  ).T.reshape((self.W).shape)
        self.dldb=np.sum(dloss,axis=(0,2,3)).reshape(self.num_filters,1) #NFHW F*1
        self.dldx_Col=fdloss @ self.W_col.T
        self.dldx=im2col_bw(self.dldx_Col, self.inputs.shape, self.k_height, self.k_width, pad = self.pad, stride = self.stride)
        return self.dldw,self.dldb,self.dldx

    
    def update(self, learning_rate=0.001, momentum_coeff=0.9):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as Problem1
        Here we divide gradients by batch_size (because we will be using sum Loss
        instead of mean Loss in Problem 2 during backpropogation). Do not divide
        gradients by batch_size in step() in Problem 1.
        """
        
        self.dldb_momentum_new =self.dldb_momentum * momentum_coeff + self.dldb / self.batch_size
        self.b = self.b - learning_rate * self.dldb_momentum_new
        self.dldw_momentum_new = self.dldw_momentum * momentum_coeff + self.dldw / self.batch_size
        self.W= self.W - learning_rate * self.dldw_momentum_new 
        self.dldb_momentum=self.dldb_momentum_new
        self.dldw_momentum=self.dldw_momentum_new
        
        
    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.W, self.b
    
    '''
    def zerograd(self):
        self.dldw=np.zeros((self.num_filters,self.C, self.k_height, self.k_width))
        self.dldx=np.zeros((self.C, self.num_filters, self.H, self.Width))
        self.dldb=np.zeros(self.b.shape)
    '''

class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filter_height=filter_shape[0]
        self.filter_width=filter_shape[1]
        self.stride=stride

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (N, C, H, W)
        
        self.inputs=inputs
        self.shape = inputs.shape
        N,C,H,W = inputs.shape
        self.height_wave = 1 + (H - self.filter_height) // self.stride
        self.width_wave = 1 + (W - self.filter_width) // self.stride
        inputs_reshaped = inputs.reshape(N * C, 1, H, W)
        self.im2col_inputs = im2col(inputs_reshaped, self.filter_height, self.filter_width, stride=self.stride,padding=0)
        self.max_idx = np.argmax(self.im2col_inputs,0)
        self.output = self.im2col_inputs[self.max_idx, range(self.max_idx.size)]
        self.output = self.output.reshape(self.height_wave, self.width_wave, N, C)
        self.output = self.output.transpose(2, 3, 0, 1)
        return self.output
        """
        self.inputs=inputs
        N, C, H, W = inputs.shape
        out_h = int(1 + (H - self.filter_height) // self.stride)
        out_w = int(1 + (W - self.filter_width) // self.stride)
        col = im2col(inputs, self.filter_height, self.filter_width, stride=self.stride, pad=0)
        col = col.reshape(-1, self.filter_height*self.filter_width)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.arg_max = arg_max

        return out
    
    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()

        N,C,H,W = dloss.shape
        dmax = np.zeros(self.shape)
        for n in range(N):
            for c in range(C):
                for h in range(self.height_wave):
                    for w in range(self.width_wave):
                        h1 = h * self.stride
                        h2 = h * self.stride + self.filter_height
                        w1 = w * self.stride
                        w2 = w * self.stride + self.filter_width
                        window = self.inputs[n, c, h1:h2, w1:w2]
                        window2 = np.reshape(window, (self.filter_height*self.filter_width))
                        window3 = np.zeros_like(window2)
                        window3[np.argmax(window2)] = 1
                        dmax[n,c,h1:h2,w1:w2] = np.reshape(window3,(self.filter_height,self.filter_width)) * dloss[n,c,h,w]
        return dmax
        """
        dloss = dloss.transpose(0, 2, 3, 1) #N h, W, C
        
        pool_size = self.filter_height * self.filter_width
        dmax = np.zeros((dloss.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dloss.flatten()
        dmax = dmax.reshape(dloss.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = im2col_bw(dcol, self.inputs.shape, self.filter_height, self.filter_width, stride=self.stride, pad=0)
        
        return dx

    
class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        b = np.sqrt(6) / np.sqrt(indim + outdim)
        self.W = np.random.uniform(-b, b, (indim, outdim))
        self.b = np.zeros((outdim, 1))
        self.dldw_momentum=0
        self.dldb_momentum=0
        self.indim=indim
        self.outdim=outdim
        #self.dldx=np.zeros(self.indim)
        #self.dldw=np.zeros((self.indim,self.outdim))
        #self.dldb=np.zeros(self.outdim)

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        self.inputs=inputs
        self.h = np.matmul(inputs,self.W) + self.b.T
        return self.h

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        self.dldx = np.matmul(dloss, self.W.T)
        self.dldw = np.matmul(self.inputs.T, dloss)
        self.dldb= np.sum(dloss, axis = 0, keepdims=True).T

        return self.dldw,self.dldb,self.dldx

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        self.batch_size=self.inputs.shape[0]
        self.dldw_momentum_new=momentum_coeff*self.dldw_momentum + self.dldw/self.batch_size
        self.W= self.W - learning_rate * self.dldw_momentum_new
        self.dldb_momentum_new=momentum_coeff*self.dldb_momentum + self.dldb/self.batch_size
        self.b= self.b -learning_rate *  self.dldb_momentum_new
        self.dldb_momentum=self.dldb_momentum_new
        self.dldw_momentum=self.dldw_momentum_new

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.W, self.b
    
    '''
    def zerograd(self):
        self.dldx=np.zeros(self.indim)
        self.dldw=np.zeros((self.indim,self.outdim))
        self.dldb=np.zeros(self.outdim)
    '''

class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        self.labels=labels
        self.logits=logits
        self.softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        cross_entropy = np.sum(-np.sum(labels*np.log(self.softmax),axis=1), axis=0)
        return cross_entropy

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        dldo=self.softmax-self.labels
        return dldo

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        pass



class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        self.flat=Flatten()
        self.layers=[]
        self.layers.append(Conv((3,32,32),(5,5,5),rand_seed=0))
        self.layers.append(ReLU())
        self.layers.append(MaxPool((2,2), stride=2))
        self.layers.append(LinearLayer(1280,10,rand_seed=0))
        self.layers.append(SoftMaxCrossEntropyLoss())

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        self.inputs=inputs
        self.y_labels=y_labels
        inputs_1=self.layers[0].forward(self.inputs, stride=1, pad=2)
        inputs_2= self.layers[1].forward(inputs_1)
        inputs_3= self.layers[2].forward(inputs_2)
        inputs_flat_3= self.flat.forward(inputs_3)
        self.inputs_out= self.layers[3].forward(inputs_flat_3)
        self.pred=self.inputs_out.argmax(axis=-1)
        self.loss=self.layers[-1].forward(self.inputs_out, y_labels)
        return self.loss, self.pred
    
    def get_Acc (self, y_labels):
        y=[np.where(r==1)[0][0] for r in y_labels]
        acc= sum(self.pred==y)/len(y_labels)
        return acc
    
    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        self.dld_5=self.layers[4].backward()
        self.dld_4=self.layers[3].backward(self.dld_5)[2]
        self.dld_4_noflat=self.flat.backward(self.dld_4)
        self.dld_3=self.layers[2].backward(self.dld_4_noflat)
        self.dld_2=self.layers[1].backward(self.dld_3)
        self.dld_1=self.layers[0].backward(self.dld_2)

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.layers[3].update(learning_rate,momentum_coeff)
        self.layers[0].update(learning_rate,momentum_coeff)
        
    def zerograd(self):
        for layer in self.layers:
            if hasattr(layer, 'zerograd'):
                layer.zerograd()

class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 5x5x5
        then apply Relu
        then Conv with filter size of 5x5x5
        then apply Relu
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        self.flat=Flatten()
        self.layers=[]
        self.layers.append(Conv((3,32,32),(5,5,5),rand_seed=0))
        self.layers.append(ReLU())
        self.layers.append(MaxPool((2,2), stride=2))
        self.layers.append(Conv((5,16,16),(5,5,5),rand_seed=0))
        self.layers.append(ReLU())
        self.layers.append(Conv((5,16,16),(5,5,5),rand_seed=0))
        self.layers.append(ReLU())
        self.layers.append(LinearLayer(1280,10,rand_seed=0))
        self.layers.append(SoftMaxCrossEntropyLoss())

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        self.inputs=inputs
        self.y_labels=y_labels
        inputs_1=self.layers[0].forward(self.inputs, stride=1, pad=2)
        inputs_2= self.layers[1].forward(inputs_1)
        inputs_3= self.layers[2].forward(inputs_2)
        inputs_4=self.layers[3].forward(inputs_3, stride=1, pad=2)
        inputs_5= self.layers[4].forward(inputs_4)
        inputs_6=self.layers[5].forward(inputs_5, stride=1, pad=2)
        inputs_7= self.layers[6].forward(inputs_6)
        inputs_flat_7= self.flat.forward(inputs_7)
        self.inputs_out= self.layers[7].forward(inputs_flat_7)
        self.pred=self.inputs_out.argmax(axis=-1)
        self.loss=self.layers[-1].forward(self.inputs_out, y_labels)
        return self.loss, self.pred


    def get_Acc (self, y_labels):
        pred_hot=labels2onehot(self.pred)
        acc= sum(pred_hot==y_labels)/len(y_labels)
        return acc
    
    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        dld_9=self.layers[8].backward()
        dld_8=self.layers[7].backward(dld_9)[2]
        dld_8_noflat=self.flat.backward(dld_8)
        dld_7=self.layers[6].backward(dld_8_noflat)
        dld_6=self.layers[5].backward(dld_7)[2]
        dld_5=self.layers[4].backward(dld_6)
        dld_4=self.layers[3].backward(dld_5)[2]
        dld_3=self.layers[2].backward(dld_4)
        dld_2=self.layers[1].backward(dld_3)
        self.dld_1=self.layers[0].backward(dld_2)[2]

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.layers[7].update(learning_rate,momentum_coeff)
        self.layers[5].update(learning_rate,momentum_coeff)
        self.layers[3].update(learning_rate,momentum_coeff)
        self.layers[0].update(learning_rate,momentum_coeff)
    def zerograd(self):
        for layer in self.layers:
            if hasattr(layer, 'zerograd'):
                layer.zerograd()

class MLP:
    """
    Implement as you wish, not autograded
    """
    def __init__(self):
        pass

    def forward(self, inputs, y_labels):
        pass

    def backward(self):
        pass

    def update(self,learning_rate,momentum_coeff):
        pass


# Implement the training as you wish. This part will not be autograded.
if __name__ == '__main__':
    # This part may be helpful to write the training loop
    #from load_cifar import trainX, trainy, testX, testy
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    # Training parameters
    parser = ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--momentum', type=float, default = 0.9)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--conv_layers', type=int, default = 3)
    parser.add_argument('--filters', type=int, default = 1)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    #print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    train_data = trainX.reshape(-1,3,32,32).astype(np.float32)/255.0
    train_label = np.array([[i==lab for i in range(10)] for lab in trainy], np.int32)
    test_data = testX.reshape(-1,3,32,32).astype(np.float32)/255.0
    test_label = np.array([[i==lab for i in range(10)] for lab in testy], np.int32)
    train_log=[]
    test_log=[]
    losstrain_log=[]
    losstest_log=[]
    num_train = len(train_data)
    num_test = len(test_data)
    batch_size = 128
    train_iter = num_train//batch_size + 1
    test_iter = num_test//batch_size + 1
    if args.conv_layers == 1:
        cnn = ConvNet()
    elif args.conv_layers == 3:
        cnn = ConvNetThree()
    start = datetime.now()
    for epoch in range(100):
        idxs = np.random.permutation(num_train)
        batches=[]
        for start_idx in range(0, num_train, batch_size):
            if start_idx+batch_size > num_train:
                batch_idx = idxs[start_idx: num_train]
                batches.append([train_data[batch_idx], train_label[batch_idx],trainy[batch_idx]])
            else:
                batch_idx = idxs[start_idx: start_idx + batch_size]
                batches.append([train_data[batch_idx], train_label[batch_idx],trainy[batch_idx]])
        train_acc=[]
        train_los=[]
        train_preds=[]
        lossall=0
        for batch in batches:
            train_loss, train_pred=cnn.forward(batch[0], batch[1])
            lossall+=train_loss
            dldx=cnn.backward()
            cnn.update(learning_rate=0.01, momentum_coeff=0.9)
            #cnn.zerograd()
            #train_acc.append(sum(train_pred==batch[2])/len(batch[2]))
            #train_los.append(train_loss)
            train_preds=np.append(train_preds,train_pred)
        #losstrain_log.append(np.mean(train_los))
        #train_log.append(np.mean(train_acc))
        #train_loss,train_predict=cnn.forward(train_data,train_label)
        train_log.append(sum(train_preds==trainy[idxs])/len(train_label)) 
        losstrain_log.append(lossall)
        test_loss,test_predict=cnn.forward(test_data,test_label)
        test_log.append(sum(test_predict==testy)/len(test_label)) 
        losstest_log.append(test_loss)
    losstrain_log=[i/len(trainy)for i in losstrain_log]
    losstest_log=[i/len(testy) for i in losstest_log]
    print(datetime.now() - start)
    fig, host = plt.subplots(figsize=(8,5))
    host.set_ylim(0, 1)
    par1 = host.twinx()
    par1.set_ylim(0, max(losstest_log))
    host.set_ylabel("accuracy")
    par1.set_ylabel("loss")
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.7)
    color4 = plt.cm.viridis(.9)
    p1, = host.plot(test_log,color=color1, label="test_accuracy")
    p2, = par1.plot(losstest_log, color=color2, label="test_loss")
    p3, = host.plot(train_log,color=color3, label="train_accuracy")
    p4, = par1.plot(losstrain_log, color=color4, label="train_loss")
    lns = [p1, p2,p3,p4]
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
np.savetxt("/Users/johnson/Desktop/CNN4.csv", a, delimiter=",")






#%%
text=np.loadtxt("/Users/johnson/Desktop/CNN4.csv", delimiter=",") 
test_log, train_log,losstest_log,losstrain_log=text
fig, host = plt.subplots(figsize=(8,5))
host.set_ylim(0, 1)
par1 = host.twinx()
par1.set_ylim(0, max(losstest_log))
host.set_ylabel("accuracy")
par1.set_ylabel("loss")
color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.7)
color4 = plt.cm.viridis(.9)
p1, = host.plot(test_log,color=color1, label="test_accuracy")
p2, = par1.plot(losstest_log, color=color2, label="test_loss")
p3, = host.plot(train_log,color=color3, label="train_accuracy")
p4, = par1.plot(losstrain_log, color=color4, label="train_loss")
lns = [p1, p2,p3,p4]
host.legend(handles=lns, loc='best')
plt.show()

print(test_log[-1], train_log[-1],losstest_log[-1],losstrain_log[-1])
#%%
text=np.loadtxt("/Users/johnson/Desktop/Combine.txt") 
test1, test2,test3,test4, test5,test6,test7=text
fig, host = plt.subplots(figsize=(8,5))
host.set_ylim(0, 1)
par1 = host.twinx()
par1.set_ylim(0, max(losstest_log))
host.set_ylabel("accuracy")
color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.2)
color3 = plt.cm.viridis(0.4)
color4 = plt.cm.viridis(0.6)
color5 = plt.cm.viridis(0.7)
color6 = plt.cm.viridis(0.8)
color7 = plt.cm.viridis(0.9)
p1, = host.plot(test1,color=color1, label="SingleMLP")
p2, = host.plot(test2, color=color2, label="3MLP")
p3, = host.plot(test3,color=color3, label="BN")
p4, = host.plot(test4, color=color4, label="DP")
p5, = host.plot(test5, color=color5, label="1layer1filterCNN")
p6, = host.plot(test6, color=color6, label="1layer5filterCNN")
p7, = host.plot(test7, color=color7, label="3layer5filterCNN")
lns = [p1, p2,p3,p4, p5,p6,p7]
host.legend(handles=lns, loc='best')
plt.show()














