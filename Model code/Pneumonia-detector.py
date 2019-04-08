from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing CNN
classifier = Sequential()

#Convolution layer1
classifier.add(Convolution2D(32,3,3,input_shape=(128, 128,3),activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Convolution layer2
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#Hidden Layer
classifier.add(Dense(output_dim=128,activation='relu'))
#Output layer
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'train/',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(  'test/',
                                               target_size=(128, 128),
                                               batch_size=32,
                                               class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=5216/32,
        epochs=15,
        validation_data=test_set,
        validation_steps=624/32)

classifier.save('pneumonia-model&weights.h5')
classifier.save_weights('pneumonia-weights.h5')


training_set.class_indices