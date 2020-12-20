import DataLoader as dl
training_data, validation_data, test_data = dl.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
'''
import BasicNetwork as network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
'''
import Chap3Network as network
net = network.Network([784, 30, 10], cost = network.CrossEntropyCost())
#net.large_weight_initializer()
net.SGD(training_data[:1000], 200, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data[:100], monitor_evaluation_accuracy = True, early_stopping_n=15)
