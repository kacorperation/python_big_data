# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:38:44 2017

@author: K.Ataman
"""

class dataset_:
    """Creates input and output and batch creation method.
    Keeps track of which inputs have been used for a batch, randomizes the batch
    if most elements have been used
    """
    def __init__(self, df, train_percentage, backpropagation = 30):
        self.train_percentage = train_percentage
        self.backpropagation = backpropagation
        self.input = df[['SPOT_TENOR', 'LIMIT_TENOR', 'rfq', 'rfs', 'BUY', 'SELL', 
                   'CANCELLED', 'DONE', 'PENDING', 'RFS_PENDING', 'TRADE_DONE', 
                   'TIME_DELTA', 'CURRENCY_1_AMT', 'POS1_AMT', 'POS2_AMT',
                   'LAST_RATE', 'TRADE_RATE']]
        self.input = pd.DataFrame.as_matrix(self.input)
        
        self.output = df[['TRADE_DONE_NEXT_HOUR']]
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