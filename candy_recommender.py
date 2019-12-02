from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import pandas as pd
import numpy as np
import random
import os

class preProcessings:

    def __init__(self, X, y):

        self.X = X
        self.y = y

    @staticmethod
    def __target_class(x):

        if x <=.1:
            y = 1
        elif x <= .2:
            y = 2
        elif x <= .3:
            y = 3
        elif x <= .4:
            y = 4
        elif x <=.5:
            y = 5
        elif x <= .6:
            y = 6
        elif x <= .7:
            y = 7
        elif x <= .8:
            y = 8
        elif x <= .9:
            y = 9
        else: y = 0
            
        return y

    def target_to_categorical(self):
        
        print("Transform target into equal sized classes and fit them into one hot vector")
        y = self.y
        target = list(map(lambda x: self.__target_class(x/100), y))
        # Convert labels to categorical one-hot encoding
        one_hot_target = to_categorical(target , num_classes=10)

        return one_hot_target

    def input_to_keras(self, NUM_CLASS=9):

        print("Input format preperation for keras model")
        #cols = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy','nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus']
        cols = self.X.columns
        NUM_CLASS = cols.__len__()

        X_array = np.zeros((self.X.__len__(), NUM_CLASS))

        for idx, row in self.X.iterrows():
            values = []
            
            for col in cols:
                values.append(row[col])
            
            X_array[idx, : ] = values  

        return X_array
            

class shuffling:

    def shuffling_train_test_split(X, y, test_size=0.18):
        NUM_TEST = round(int((test_size) * X.shape[0]) + 1)
        print("Spilt randomly: ", NUM_TEST)
        idxs = np.arange(0, X.__len__())
        idxs_test = random.sample(set(idxs), NUM_TEST)
        idxs_train = [idx for idx in idxs if idx not in idxs_test]
        
        X_test = np.array([X[i] for i in idxs_test])
        X_train = np.array([X[i] for i in idxs_train])
        y_test = np.array([y[i] for i in idxs_test])
        y_train = np.array([y[i] for i in idxs_train])
        
        return X_train, X_test, y_train, y_test

    def sample_wr(population, k):
        print("Chooses k random elements (with replacement) from a population")
        n = len(population)
        _random, _int = random.random, int  # speed hack 
        result = [None] * k
        for i in range(k):
            j = _int(_random() * n)
            result[i] = population[j]
        
        return result


class neural_network_keras :
    
    def __init__(self, input_dim, target_dim):
        
        self.input_dim = input_dim
        self.target_dim = target_dim
            
        # Create a 2 layer deep neural network   
        net = Sequential()
        net.add(Dense(16, input_dim = input_dim, activation = "relu"))
        net.add(Dense(32, activation = "relu"))
        net.add(Dropout(0.5))
        net.add(Dense(self.target_dim, activation='softmax'))
        net.compile(loss = 'categorical_crossentropy' #"mean_squared_error"
                    ,optimizer='sgd', metrics=['accuracy'] #optimizer = Adam(lr = 0.01)
                )
        self.net = net

    
    def train(self, X_train, y_train, epochs, batch_size, verbose):
        # Train Model
        self.net.fit(X_train, y_train , epochs=epochs, batch_size=batch_size, verbose=verbose)


    def save(self, name):
        if not os.path.exists("data/model/"):
            os.makedirs("data/model/")
        
        self.net.save("data/model/{}".format(name))


class RecommanderModel:

    def __init__(self):

        # Load model
        MODEL_NAME = "recommender_model.h5"
        PATH_IMP = "data/model/"
        print("load modle from: ", PATH_IMP)
        self.recommender_model = load_model(PATH_IMP + MODEL_NAME)


    def recommend(self, *args):
        
        #create input vector
        X_pred = np.zeros(9)
        properties = args[0]
        X_pred[0] = 1 if "chocolate" in properties else 0
        X_pred[1] = 1 if "fruity" in properties else 0
        X_pred[2] = 1 if "caramel" in properties else 0
        X_pred[3] = 1 if "peanutyalmondy" in properties else 0
        X_pred[4] = 1 if "nougat" in properties else 0
        X_pred[5] = 1 if "crispedricewafer" in properties else 0
        X_pred[6] = 1 if "hard" in properties else 0
        X_pred[7] = 1 if "bar" in properties else 0
        X_pred[8] = 1 if "pluribus" in properties else 0
        X_pred = np.array([X_pred])
        print("X_vec = ", X_pred)

        y_pred = self.recommender_model.predict_classes(X_pred)

        print("Matchup win expectation between: {}% and {}% .".format((y_pred[0] -1)*10, y_pred[0]*10)) 

        return [(y_pred[0] -1)*10 , y_pred[0]*10]