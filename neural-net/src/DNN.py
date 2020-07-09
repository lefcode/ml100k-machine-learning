import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
#from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from math import sqrt
#import data
df = pd.read_csv('data.csv')
x_columns = df.columns
x = df['user_id'].values #users column
###One hot encoding of input
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(x)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
x=onehot_encoded  #one-hot encoded users
###Normalization of data to range [0,1]
y_cols = x_columns.drop('user_id')
my_y = df[y_cols].values
y=MinMaxScaler().fit_transform(my_y)
kf = KFold(5, shuffle=True) # Use for 5-Fold classificatioν
fold = 0
rmseList = []
maeList=[]
for train, test in kf.split(x):
    fold+=1
    print(f"Fold #{fold}")
    x_train,y_train,val_x,val_y = x[train],y[train],x[test],y[test]
    model=Sequential()
    model.add(Dense(5,input_shape=(943,),activation='relu'))
    model.add(Dense(10,input_dim=5,activation='relu'))
    model.add(Dense(20,input_dim=10,activation='relu'))
    model.add(Dense(1682,activation='linear'))
    
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
  
    opt=keras.optimizers.SGD(lr=0.1, momentum=0.6, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=opt,metrics=[rmse,'mae'])
    history = model.fit(x[train], y[train], validation_data=(val_x,val_y)
                     ,epochs=30,batch_size=10, verbose=0)
    # Evaluate model
    pred = model.predict(val_x)
    test_mae=mean_absolute_error(val_y,pred)
    print("Mean absolute error (MAE):      %f" % test_mae)
    maeList.append(test_mae)
    test_rmse=sqrt(mean_squared_error(val_y,pred))
    print("Root mean squared error (RMSE): %f" % test_rmse)
    rmseList.append(test_rmse)

print("RMSE: ", np.mean(rmseList)) #total mean
print("MAE: ", np.mean(maeList)) #total mean
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Model loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Test'], loc='upper right')
pyplot.show()