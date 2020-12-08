import DataLoader as dl
training_data, validation_data, test_data = dl.load_data_wrapper()
training_data = list(training_data)
import BasicNetwork as network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 1.0, test_data=test_data)