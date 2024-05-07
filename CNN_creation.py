import tensorflow as tf
import numpy as np
import math

def form_CNN_parameters(state, spaces):
    layers_count = math.floor(len(state)/4)
    CNN_params = {
                "CNN_count": layers_count,
                "CNN_parameters":[]
    }
    for layer in range(layers_count):
        CNN_params["CNN_parameters"].append({"kernel":spaces[0+layer*4][state[0+layer*4]], 
                                              "max_pooling": spaces[1+layer*4][state[1+layer*4]], 
                                              "units": spaces[2+layer*4][state[2+layer*4]],
                                              "activation": spaces[3+layer*4][state[3+layer*4]]})
    return CNN_params
    

def CreateModel(parameters: dict, input_shape: tuple, output_shape: tuple):
    #print(parameters)
    input_X = tf.keras.Input(shape=input_shape, name='CNN_input_X')
    CNN_count = parameters["CNN_count"]
    CNN_parameters = parameters['CNN_parameters']
    CNN = []
    MP = []
    if CNN_count >= 1:
        CNN.append(tf.keras.layers.Conv1D( filters = CNN_parameters[0]['units']
                                          , kernel_size = [CNN_parameters[0]['kernel']]
                                          , activation = CNN_parameters[0]['activation']
                                          , name=f'CNN_{1}'
                                         )(input_X))
    else:
        CNN.append(input_X)
    if CNN_count >= 1 and CNN_parameters[0]['max_pooling'] > 1:
        try:
            MP.append(tf.keras.layers.MaxPooling1D(pool_size = [CNN_parameters[0]['max_pooling']]
                                                       , name=f'MaxPool_{1}')(CNN[-1]))
        except ValueError:
            MP.append(CNN[-1])
    else:
        MP.append(CNN[-1])
    if CNN_count > 1:
        for CNN_layer in np.arange(CNN_count-1) + 1:
            CNN_layer_number = CNN_layer + 1
            CNN.append(tf.keras.layers.Conv1D( filters = CNN_parameters[CNN_layer]['units']
                                          , kernel_size = [CNN_parameters[CNN_layer]['kernel']]
                                          , activation = CNN_parameters[CNN_layer]['activation']
                                          , name=f'CNN_{CNN_layer_number}'
                                         )(MP[-1]))
            if CNN_parameters[CNN_layer]['max_pooling'] > 1:
                try:
                    MP.append(tf.keras.layers.MaxPooling1D(pool_size = [CNN_parameters[CNN_layer]['max_pooling']]
                                                       , name=f'MaxPool_{CNN_layer_number}'
                                                      )(CNN[-1]))                
                except ValueError:
                    MP.append(CNN[-1])
            else:
                MP.append(CNN[-1])
    Flat = tf.keras.layers.Flatten(name='Flat')(MP[-1])
    outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='CNN_output_y')(Flat)
    model = tf.keras.models.Model(inputs=input_X
                                      , outputs=outputs)
    return model