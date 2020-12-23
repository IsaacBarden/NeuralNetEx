import DataLoader as dl

'''
def testTheano():
    from theano import function, config, shared, sandbox
    import theano.tensor as T
    import numpy as np
    from numpy.random import RandomState
    import time
    print("Testing Theano library...")
    vlen = 10 * 30 * 768 # 10 x number of cores x number of threads per core 
    iters = 1000
    rng = RandomState(22)
    x = shared(np.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print(f"looping {iters} times took {t1-t0} seconds")
    print(f"Result is {r,}")
    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print("Used the CPU")
    else:
        print("Used the GPU")
testTheano()
'''
'''
training_data, validation_data, test_data = dl.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
'''
'''
import BasicNetwork as network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
'''
'''
import ImprovedNetwork as network
net = network.Network([784, 30, 10], cost = network.CrossEntropyCost())
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy = True, early_stopping_n=10)
'''

import theano
import numpy as np

import DeepNetwork
from DeepNetwork import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

training_data, validation_data, test_data = DeepNetwork.load_data_shared()
mini_batch_size = 10

'''
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
'''


net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2,2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2,2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], 
    mini_batch_size
)


net.SGD(training_data, 40, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1, early_stopping_n=10)
