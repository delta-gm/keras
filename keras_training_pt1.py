import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


num_classes=2
image_resize=224
batch_size=100

data_generator=ImageDataGenerator(preprocessing_function=preprocess_input,)

train_generator=data_generator.flow_from_directory('concrete_data_week4/train',
                                                   target_size=(image_resize,image_resize),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

validation_generator=data_generator.flow_from_directory('concrete_data_week4/valid',
                                                        target_size=(image_resize,image_resize),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model=Sequential()

model.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
steps_per_epoch_training=len(train_generator)
steps_per_epoch_validation=len(validation_generator)
epochs=1

fit_history=model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)

model.save('classifier_vgg16_model.h5')