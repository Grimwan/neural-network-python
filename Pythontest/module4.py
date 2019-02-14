
from keras.layers.core import Dense, Dropout, TimeDistributedDense, Reshape, Activation

NUM_LABLES=4
NUM_EXAMPLES=1280

# One-hotting labels
inds = np.random.randint(0, NUM_LABLES-1, size=(NUM_EXAMPLES, 1500))
labels = np.zeros((NUM_EXAMPLES, 1500, NUM_LABLES))

i = np.arange(NUM_EXAMPLES)
j = np.arange(1500)
ii,jj = np.ix_(i,j)

labels[ii,jj,inds] = 1

# For simplicity we assume the images are flattened.
X = np.random.randn(NUM_EXAMPLES, 1500)

mlp = Sequential()
# Your model goes here
# Assume it ends with a Fully-Connected layer
mlp.add(Dense(512, input_shape=(X.shape[1], ), activation='relu'))

# Now we predict
mlp.add(Dense(1500 * NUM_LABLES))
mlp.add(Reshape((1500, 21)))
mlp.add(Activation('softmax'))

mlp.compile(Adam(), 'categorical_crossentropy')