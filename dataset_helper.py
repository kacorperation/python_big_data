"""
TODO:
    Take care of batch creation when backpropagation = 0
    Normalization, softerization etc, specified per feature
	 Training and testing set data selection methods 
	 Batch data selection methods (epoch, batch, mini-batches etc)
"""


import pandas as pd
import numpy as np

class dataset_helper:
    """Creates input and output and batch creation method.
    Keeps track of which inputs have been used for a batch, randomizes the batch
    if most elements have been used
	
	inputs:
		df: a dataframe containing both inputs and outputs
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
	
	attributes:
		self.train_percentage:	which percent of the data will be used for training. For example,
								train_percentage = 0.9 means 90% of the data will be used for 
								training, and 1 - 0.9 = 0.1 = 10% of the data will be used
								for testing
								
		self.backpropagation:	How many time steps back does an output roughly depend on? Set this
								to 0 if the output is time invariant
                                
      self.num_features_input:Number of features of the input                          
		
		self.input_matrix:				Matrix of shape [element, self.num_features_input] 
                            that holds inputs
		
		
        
      self.num_features_output:Number of features of the output
		
		self.output_matrix:	          Matrix of shape [element, self.num_features_output] 
                            that holds outputs
		
		
		
		self.num_inputs: 		How many elements the dataset has
		
		self.num_inputs_train:	How many elements are in the training set
		
		self.num_inputs_test:	How many elements are in the testing set
    """
    def __init__(self, df, input_features, output_features, 
	train_percentage, backpropagation = 0):
        self.train_percentage = train_percentage
        self.backpropagation = backpropagation
        self.input_matrix = df[input_features]
        self.input_matrix = pd.DataFrame.as_matrix(self.input_matrix)
        
        self.output_matrix = df[output_features]
        self.output_matrix = pd.DataFrame.as_matrix(self.output_matrix)
        #force output to be 0 or 1, useful for cross entropy and classification
        
        
        self.num_inputs  = self.input_matrix.shape[0]
        self.num_features_input = self.input_matrix.shape[1]
        self.num_features_output = self.output_matrix.shape[1]
        self.num_inputs_train = int(self.train_percentage * self.num_inputs)
        self.num_inputs_test  = self.num_inputs - self.num_inputs_train - self.backpropagation - 1
        
    def training_and_testing_elements_picker(self, training_data_selection_method = 'random'):
        """
        Creates an array of indices to be used for training and testing 
        inputs and outputs depending on the training_data_selection_method
        
        current training_data_selection_methods are:
            
            random: Randomly picks training and testing elements from input
                    and output sets
            first: Selects the first self.num_inputs_train elements to be the training set
            
            last: Selects the last self.num_inputs_train elements to be the training set
            
        other variables:
            
            self.training_indices:	    An array of size [self.num_inputs_train] 
                                        that holds the indices used for training
		
            self.testing_indices:	    An array of size [self.num_inputs_test] 
                                        that holds the indices used for testing
		
        		self.training_index:     Index for self.training_indices
		
        		self.testing_index:		Index for self.testing_indices
            
        """
        data_indices = np.linspace(0, self.num_inputs - 1, self.num_inputs)
        if training_data_selection_method == 'random':
            data_indices = np.random.choice(data_indices, self.num_inputs, replace = False)
            #training_set is indices of elements self.input_matrix & 
            #self.output_matrix that will be used for training
            self.training_indices = data_indices[0:self.num_inputs_train]
            self.testing_indices  = data_indices[self.num_inputs_train : self.num_inputs_train + self.num_inputs_test]
        elif training_data_selection_method == 'first':
            self.training_indices = data_indices[0:self.num_inputs_train]
            self.testing_indices  = data_indices[self.num_inputs_train : self.num_inputs]
        elif training_data_selection_method == 'last':
            self.training_indices = data_indices[self.num_inputs-self.num_inputs_train:self.num_inputs]
            self.testing_indices  = data_indices[0 : self.num_inputs-self.num_inputs_train]
        
        #indexes to determine which training and testing variables we are at
        self.training_index = 0
        self.testing_index  = 0
    def create_batch(self, batch_size, training = True):
        """creates a batch for training, with the shapes
        output_x = [batch_size, self.backpropagation, self.num_features_input] if backprop > 0
        output_x = [batch_size, self.num_features_input] if backprop = 0
        output_y = [batch_size, self.num_features_output]
        """
        """
        Reminder:
            Epoch:
                Algorithm sees every training instance exactly once
            Batch:
                Using all data for one iteration (batch_size = num_elements)
            Mini-batch:
                If not using all data in one iteration, then what we have are
                "mini-batches" (batch_size < num_elements)
        """
        if self.backpropagation == 0:
            output_x = np.zeros([batch_size, self.num_features_input])
        else:
            output_x  = np.zeros([batch_size, self.backpropagation, self.num_features_input])
        
        output_y = np.zeros([batch_size, self.num_features_output])
        
        #are we training?
        if training == True:
            # batch_size < num_inputs_train
            counter = 0
            while counter < batch_size:
                #if batch exceeds the number of elements, stop creating batches. 
                #program will return zeros for elements after epoch is complete
                if self.training_index + counter + self.backpropagation > self.num_inputs_train:
                    self.training_index = 0
                    break
                for i in range(batch_size):
                    self.training_index = i
                    output_y[i] = self.output_matrix[self.training_indices[self.training_index]]
                    for j in range(self.backpropagation):
                        if self.backpropagation > 0:
                            output_x[i, j] = self.input_matrix[self.training_indices[self.training_index]]
                        else:
                            output_x[i] = self.input_matrix[self.training_indices[self.training_index]]
                        self.training_index +=1
                counter += 1

        #we are testing        
        else:
            counter = 0
            while counter < batch_size:
                #if batch exceeds the number of elements, stop creating batches. 
                #program will return zeros for elements after epoch is complete
                if self.testing_index + counter + self.backpropagation > self.num_inputs_testing:
                    self.testing_index = 0
                    break
                for i in range(batch_size):
                    self.testing_index = i
                    output_y[i] = self.output_matrix[self.testing_indices[self.testing_index]]
                    for j in range(self.backpropagation):
                        if self.backpropagation > 0:
                            output_x[i, j] = self.input_matrix[self.testing_indices[self.testing_index]]
                        else:
                            output_x[i] = self.input_matrix[self.testing_indices[self.testing_index]]
                        self.testing_index +=1
                counter += 1
                
        return output_x, output_y
