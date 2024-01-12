EPOCH_NUMBER = 200 # best result
BATCH_SIZE_NUMBER = 64 # best result, ask for an info about batch size
COST_FUNCTION = 'mean_absolute_error'
LEARNING_RATE_ADAM = 0.001
ACTIVATION_RELU = 'relu'
ACTIVATION_LEAKY_RELU = 'LeakyReLU'
ACTIVATION_TANH = 'tanh' 
ACTIVATION_LINEAR = 'linear'
VERBOSE = 1
TEST_SIZE = 0.2
FIRST_AND_THIRD_LAYER_NEURONS = 128 
SECOND_LAYER_NEURONS = 256
FOURTH_LAYER_NEURONS = 1 

class ModelConfiguration():
    def __init__(self):
        self.epoch_number = EPOCH_NUMBER
        self.batch_size_number = BATCH_SIZE_NUMBER
        self.cost_function = COST_FUNCTION
        self.activation_relu = ACTIVATION_RELU        
        self.activation_leaky_relu = ACTIVATION_LEAKY_RELU        
        self.activation_tanh = ACTIVATION_TANH        
        self.activation_linear = ACTIVATION_LINEAR        
        self.number_of_neurons_in_first_and_third_hidden_layer = FIRST_AND_THIRD_LAYER_NEURONS
        self.number_of_neurons_in_second_hidden_layer = SECOND_LAYER_NEURONS
        self.number_of_neurons_in_fourth_hidden_layer = FOURTH_LAYER_NEURONS
        self.verbose = VERBOSE
        self.learning_rate_adam = LEARNING_RATE_ADAM
        

    @property
    def preprocessed_df(self):
        return self._preprocessed_df

    @property
    def epoch_number(self):
        return self._epoch_number

    @epoch_number.setter
    def epoch_number(self, value):
        self._epoch_number = value

    @property
    def batch_size_number(self):
        return self._batch_size_number

    @batch_size_number.setter
    def batch_size_number(self, value):
        self._batch_size_number = value

    @property
    def cost_function(self):
        return self._cost_function

    @cost_function.setter
    def cost_function(self, value):
        self._cost_function = value

    @property
    def activation_relu(self):
        return self._activation_relu

    @activation_relu.setter
    def activation_relu(self, value):
        self._activation_relu = value

    @property
    def activation_leaky_relu(self):
        return self._activation_leaky_relu

    @activation_leaky_relu.setter
    def activation_leaky_relu(self, value):
        self._activation_leaky_relu = value

    @property
    def activation_tanh(self):
        return self._activation_tanh

    @activation_tanh.setter
    def activation_tanh(self, value):
        self._activation_tanh = value

    @property
    def activation_linear(self):
        return self._activation_linear

    @activation_linear.setter
    def activation_linear(self, value):
        self._activation_linear = value

    @property
    def neurons_in_first_and_third_layer(self):
        return self._neurons_in_first_and_third_layer

    @neurons_in_first_and_third_layer.setter
    def neurons_in_first_and_third_layer(self, value):
        self._neurons_in_first_and_third_layer = value

    @property
    def neurons_in_second_layer(self):
        return self._neurons_in_second_layer

    @neurons_in_second_layer.setter
    def neurons_in_second_layer(self, value):
        self._neurons_in_second_layer = value

    @property
    def neurons_in_fourth_layer(self):
        return self._neurons_in_fourth_layer

    @neurons_in_fourth_layer.setter
    def neurons_in_fourth_layer(self, value):
        self._neurons_in_fourth_layer = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def learning_rate_adam(self):
        return self._learning_rate_adam

    @learning_rate_adam.setter
    def learning_rate_adam(self, value):
        self._learning_rate_adam = value
     
    