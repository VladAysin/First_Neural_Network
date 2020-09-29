import tensorflow as tf
import numpy as np
import pandas as pd
import csv




df = pd.read_csv('iris.csv')

test_dataset = df[df['species']=='setosa'].loc[:,'sepal_length':'petal_width'].head(10).append(df[df['species']=='versicolor'].loc[:,'sepal_length':'petal_width'].head(10)).append(df[df['species']=='virginica'].loc[:,'sepal_length':'petal_width'].head(10)).reset_index(drop
= True).values.tolist()
test_labels = df[df['species']=='setosa'].loc[:,'species'].head(10).append(df[df['species']=='versicolor'].loc[:,'species'].head(10)).append(df[df['species']=='virginica'].loc[:,'species'].head(10)).reset_index(drop= True)
train_dataset = df[df['species']=='setosa'].loc[:,'sepal_length':'petal_width'].tail(40).append(df[df['species']=='versicolor'].loc[:,'sepal_length':'petal_width'].tail(40)).append(df[df['species']=='virginica'].loc[:,'sepal_length':'petal_width'].tail(40)).reset_index(drop
= True).values.tolist()
train_labels = df[df['species']=='setosa'].loc[:,'species'].tail(40).append(df[df['species']=='versicolor'].loc[:,'species'].tail(40)).append(df[df['species']=='virginica'].loc[:,'species'].tail(40)).reset_index(drop= True)

test_labels = test_labels.map({'setosa':[1,0,0],'versicolor':[0,1,0],'virginica':[0,0,1]}).values.tolist()
train_labels = train_labels.map({'setosa':[1,0,0],'versicolor':[0,1,0],'virginica':[0,0,1]}).values.tolist()


train_dataset = np.array(train_dataset,dtype=np.float)
train_labels = np.array(train_labels)



model = tf.keras.Sequential([
  tf.keras.layers.Dense(4,activation=tf.nn.softmax,use_bias=True),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  tf.keras.layers.Dense(3)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.MeanSquaredError(),
  metrics=['accuracy'])
print(len(train_dataset),len(train_labels))
model.fit(train_dataset,train_labels,epochs=70,batch_size=1,shuffle=True)

#model.evaluate(test_dataset)
prediction = model.predict(test_dataset)
count = 0
for i in range(len(prediction)):
  if (np.argmax(prediction[i])==test_labels[i].index(1)):
    count += 1
# print(count)
# print(prediction[5])
# print(test_labels[5])
# print(prediction[15])
# print(test_labels[15])
# print(prediction[25])
# print(test_labels[25])
