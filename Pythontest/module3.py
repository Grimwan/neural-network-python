import pickle 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
pickle_in = open("x.TotalTrainingData","rb")
x_train = pickle.load(pickle_in)
pickle_in = open("y.TotalTrainingData","rb")
y_train = pickle.load(pickle_in)

print("hello world")
print(x_train.shape)
#plt.imshow(x_train[0], cmap="gray")
#plt.show()
plt.imshow(x_train[0], cmap="rainbow")
plt.show()


#normalising the values for the nn to train faster
#x_train = tf.keras.utils.normalize(x_train, axis=1) # makes it between 0-1 easier to learn for network

#start of the neural network
def build_model():
    model = tf.keras.models.Sequential() # feed forward one of two
    #model.add(tf.keras.layers.Flatten()) # flattens a mutli array in to single array from 3d to 1d basically
    #First hidden layer
    model.add(tf.keras.layers.Dense(1228, activation = tf.nn.relu, input_shape=(1200,))) # tf.nn.relu = a kind of activation like sigmoid function
    #second hidden layer 
    model.add(tf.keras.layers.Dense(1228, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(1228, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(1200, activation = tf.nn.relu))
    #output layer of neural network
    model.add(tf.keras.layers.Dense(1200, activation = tf.nn.sigmoid)) 
    #nn done set up 

    #setting up how to train the neural network specifiyng optimiser and loss function
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) #lossfunction the degree of error minimise loss, optimizer is the backprogation type. keras supports alot, adam is one of them. for loss the most popular one is categorical crossentropy  
    return model
x_train = x_train.reshape(60,20*20*3)
x_train = x_train.astype('float32') / 255
y_train = y_train.reshape(60,20*20*3)
y_train = y_train.astype('float32') / 255

print(x_train[2].shape)
#print(y_train[0])
#time to train the model


import numpy as np
def KfoldValidation(train_data,train_targets,k,num_epochs):
    num_val_samples = len(train_data) // k
    all_scores = []
    for i in range(k) :
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
            train_data[(i + 1) * num_val_samples:]], axis=0)

        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
            train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        model.fit(partial_train_data, partial_train_targets,
                  epochs=num_epochs, batch_size=1, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets) #returns loss and metrics value ,verbose=0
        all_scores.append(val_mae)
    return all_scores 




#Accuracy = KfoldValidation(x_train,y_train,4,2)

#print(Accuracy)
#print(np.mean(Accuracy))
model = build_model()
model.fit(x_train,y_train, epochs = 3)
ynew = model.predict(x_train[2:3])

x_train = x_train.astype('float32') * 255
ynew = ynew.astype('float32') * 255
x_train = x_train.reshape(60,20,20,3)
ynew = ynew.reshape(1,20,20,3)

plt.imshow(ynew[0], cmap="rainbow")
plt.show()