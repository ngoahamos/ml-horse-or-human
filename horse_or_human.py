import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.preprocessing import image
import scipy
import numpy as np

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

train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(training_dir, batch_size=128, target_size=(300,300), class_mode = 'binary')

validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=128, target_size=(300,300), class_mode = 'binary')


max_pooling = MaxPooling2D(2,2)

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

def predit_image(image_path):
    print('About to predict {}'.format(image_path))
    img = image.load_img('test1.jpg', target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    clases = model.predict(image_tensor)

    print(clases)
    print(clases[0])

    if (clases[0] > 0.5):
        print('{} image is a human'.format(image_path))
    else:
        print('{} image is a horse'.format(image_path))
    
images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
for image_path in images:
    predit_image(image_path)

