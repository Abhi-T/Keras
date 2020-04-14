#Simple linear regression using Keras
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import initializers
from keras import optimizers
import matplotlib.pyplot as plt

#set the mean, STD, size of data
mu, sigma, size= 0, 4, 100

#set the slope and intercept
m , b =2, 100

#create dataset, use numpy function
x=np.random.uniform(1,10,size)
df=pd.DataFrame({'x':x})

#create Y values, using standard formula
df['y_perfect']=df['x'].apply(lambda x:m*x+b)

#create some noise as well and add it to output y
df['noise']=np.random.normal(mu,sigma,size=(size,))
df['y']=df['y_perfect']+df['noise']

#use matplotlib to visualize the data
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.plot(df['x'],df['y'],color='red')
plt.scatter(df['x'],df['y'],color='blue')
plt.show()

from keras.callbacks import Callback

class PrintAndSaveWeights(Callback):
    def on_train_begin(self, logs={}):
        self.weights_history={"m":[],"b":[]}

    def on_epoch_begin(self, batch, logs={}):
        current_m=self.model.layers[-1].get_weights()[0][0][0]
        current_b=self.model.layers[-1].get_weights()[1][0]

        self.weights_history['m'].append(current_m)
        self.weights_history['b'].append(current_b)

        print("\nm=%.2f b=%.2f\n" % (current_m, current_b))

print_save_weights = PrintAndSaveWeights()


#Create our model with a single dense layer, with a linear activation function and glorot (Xavier) input normalization

model = Sequential([
        Dense(1, activation='linear', input_shape=(1,), kernel_initializer='glorot_uniform')
# now the model will take as input arrays of shape (1,) and output arrays of shape (1,)

    ])
model.compile(loss='mse', optimizer=optimizers.sgd(lr=0.001)) ## To try our model with an Adam optimizer simple replace 'sgd' with 'Adam'

history = model.fit(x=df['x'], y=df['y'], validation_split=0.2, batch_size=1, epochs=200, callbacks=[print_save_weights])

## Save and print our final weights
predicted_m = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]
print("\nm=%.2f b=%.2f\n" % (predicted_m, predicted_b))

#visualize
plt.plot(print_save_weights.weights_history['m'])
plt.plot(print_save_weights.weights_history['b'])
plt.title('Predicted Weights')
plt.ylabel('weights')
plt.xlabel('epoch')
plt.legend(['m', 'b'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

#visualize the input data with learned parameters
df['y_predicted'] = df['x'].apply(lambda x: predicted_m*x + predicted_b)
plt.plot(df['x'],df['y_predicted'],color='blue')
plt.show()

