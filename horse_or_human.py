import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import scipy

import urllib.request
import zipfile

training_dir = "horse-or-human/training/"
validation_dir = 'horse-or-human/validation/'

# url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
# file_name = "horse-or-human.zip"
# urllib.request.urlretrieve(url, file_name)
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()

train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(training_dir, batch_size=128, target_size=(300,300), class_mode = 'binary')

validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=128, target_size=(300,300), class_mode = 'binary')

initial_cov = Conv2D(16,(3,3), activation='relu', input_shape=(300,300,3))
max_pooling = MaxPooling2D(2,2)

conv32 = Conv2D(32,(3,3), activation='relu')
conv64 = Conv2D(64,(3,3), activation='relu')

input_layer = Flatten()
hidden_layer = Dense(512,activation='relu')
output_layer = Dense(1, activation = tf.nn.sigmoid)

model = Sequential([
                Conv2D(16,(3,3), activation='relu', input_shape=(300,300,3)),
                MaxPooling2D(2,2),
                
                Conv2D(32,(3,3), activation='relu'),
                max_pooling,
                
                Conv2D(64,(3,3), activation='relu'),
                max_pooling,

                Conv2D(64,(3,3), activation='relu'), 
                max_pooling, 
                
                Conv2D(64,(3,3), activation='relu'), 
                max_pooling, 
                
                input_layer, 
                hidden_layer, 
                output_layer])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])
model.summary()
model.fit(train_generator, epochs=15, validation_data = validation_generator)
