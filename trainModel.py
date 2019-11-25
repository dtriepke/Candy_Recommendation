from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import random

import candyRecommender as cr



if __name__ == "__main__":

    # load data
    print("load data")
    FILE = "data/candy-data.csv"
    data = pd.read_csv(FILE, sep =",", encoding='latin-1')
    
    # Split by target and input
    cols = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy','nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus']
    X = data.loc[:, cols]
    y = data.loc[:, "winpercent"]

    # Pre Processings
    preProcessings = cr.preProcessings(X = X, y = y)
    X = preProcessings.input_to_keras()
    one_hot_target = preProcessings.target_to_categorical()


    # Split test and train data
    sampler = cr.shuffling

    random.seed(18)
    X_train, X_test, y_train, y_test = sampler.shuffling_train_test_split(X=X, y=one_hot_target)

    # Bootstrap train data
    print("Sample with replacement to increase the observations")
    tmp = [[x,y] for x,y in zip(X_train, y_train)]
    tmp_bootstrap = sampler.sample_wr(tmp, 300)
    X_train = np.array([tmp_bootstrap[i][0] for i in range(tmp_bootstrap.__len__())])
    y_train = np.array([tmp_bootstrap[i][1] for i in range(tmp_bootstrap.__len__())])

    print("Sampel size: ", X_train.__len__(), y_train.__len__())


    # Create the neural network for estimating the q values for choose action
    print("Create NN")
    recommender_nn = cr.neural_network_keras(input_dim  = 9, target_dim = 10)

    # Train model
    print("Train model")
    recommender_nn.train(X_train = X_train, y_train = y_train, epochs=1000, batch_size=3, verbose=1)

    # Test result
    print("Evaluation: ", recommender_nn.net.evaluate(X_test, y_test, batch_size=1) ) 

    # Save model
    print("Save model to : data/model/")
    recommender_nn.save( name = "recommender_model.h5")



