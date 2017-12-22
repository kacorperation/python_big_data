"""
TODO:
	Take care of batch creation when backpropagation = 0
"""

class dataset_helper:
    """Creates input and output and batch creation method.
    Keeps track of which inputs have been used for a batch, randomizes the batch
    if most elements have been used
	
	inputs:
	df: a dataframe
	training_elements: 	which features from dataframe will be used for inputs. Is a 
						list of strings of the form ['var_1', 'var_2, ...]
						
	output_features:	which features from dataframe will be used for outputs. Is a 
						list of strings of the form ['var_1', 'var_2, ...]
						
	train_percentage:	which percent of the data will be used for training. For example,
						train_percentage = 0.9 means 90% of the data will be used for 
						training, and 1 - 0.9 = 0.1 = 10% of the data will be used
						for testing
						
	backpropagation:	How many time steps back does an output roughly depend on? Set this
						to 0 if the output is time invariant
    """
    def __init__(self, df, input_features, output_features, 
	train_percentage, backpropagation = 30):
        self.train_percentage = train_percentage
        self.backpropagation = backpropagation
        self.input = df[input_features]
        self.input = pd.DataFrame.as_matrix(self.input)
        
        self.output = df[output_features]
        self.output = pd.DataFrame.as_matrix(self.output)
        #force output to be 0 or 1, useful for cross entropy and classification
        
        
        self.num_inputs  = self.input.shape[0]
        self.num_features_input = self.input.shape[1]
        self.num_features_output = self.output.shape[1]
        self.num_inputs_train = int(self.train_percentage * self.num_inputs)
        self.num_inputs_test  = self.num_inputs - self.num_inputs_train - self.backpropagation - 1
        
        #select data for training and testing
        data_indices = np.zeros([self.num_inputs], dtype = np.int)
        #make sure the last backpropagation elements are not selected for 
        #training and testing, as they are a pain to deal with
        for i in range (self.num_inputs - self.backpropagation - 1):
            data_indices[i] = i
        '''
        create I/O of the form
        x_train = np.zeros([num_inputs_train,backpropagation,num_features_input])
        y_train = np.zeros([num_inputs_train,num_features_output])
        x_test = np.zeros([num_inputs_test,backpropagation,num_features_input])
        y_test = np.zeros([num_inputs_test,num_features_output])
        '''
        data_indices = np.random.choice(data_indices, self.num_inputs, replace = False)
        #training_set is indices of elements self.input & 
        #self.output that will be used for training
        self.training_set = data_indices[0:self.num_inputs_train]
        self.testing_set  = data_indices[self.num_inputs_train : self.num_inputs_train + self.num_inputs_test]
        
        #indexes to determine which training and testing variables we are at
        self.training_index = 0
        self.testing_index  = 0
    def create_batch(self, batch_size, training = True):
        """creates a batch for training, where
        output_x = np.zeros([batch_size,self.backpropagation,self.num_features_input])
        output_y = np.zeros([batch_size,self.num_features_output])
        """
        #Do we want training data or testing data?
        output_x = np.zeros([batch_size,self.backpropagation,self.num_features_input])
        output_y = np.zeros([batch_size,self.num_features_output])
        if training == True:
            #do we have enough training variables left for the batch this size?
            if self.training_index + batch_size > self.num_inputs_train:
                #pick randomly
                chosen = np.random.choice(self.training_set, batch_size, replace = False)
            else:
                chosen = self.training_set[self.training_index:self.training_index + batch_size]
            for batch_element in range(batch_size):
                for backprop in range(self.backpropagation):
                    output_x[batch_element, backprop] = self.input[chosen[batch_element] + backprop]
                output_y[batch_element] = self.output[chosen[batch_element]]
            self.training_index += 1
            return output_x, output_y
        #testing mode
        else:
            if self.testing_index + batch_size > self.num_inputs_train:
                chosen = np.random.choice(self.testing_set, batch_size, replace = False)
            else:
                chosen = self.testing_set[self.testing_index:self.testing_index + batch_size]
            for batch_element in range(batch_size):
                for backprop in range(self.backpropagation):
                    output_x[batch_element, backprop] = self.input[
                        chosen[batch_element] + backprop]
                output_y[batch_element] = self.output[chosen[batch_element]]
            self.testing_index += 1
            return output_x, output_y
