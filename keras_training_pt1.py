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



from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as rn_pp_input
from keras.models import load_model
rn_model = load_model('classifier_resnet_model.h5')

rn_generator=ImageDataGenerator(
    preprocessing_function=rn_pp_input,
)

test_generator_rn=rn_generator.flow_from_directory(
    'concrete_data_week4/test',
    target_size=(image_resize,image_resize),
    shuffle=False
)

test_generator_vgg=data_generator.flow_from_directory(
    'concrete_data_week4/test',
    target_size=(image_resize,image_resize),
    shuffle=False
)



rn_score=rn_model.evaluate_generator(test_generator_rn,verbose=1)

print('Accuracy: {}% \n Loss: {}'.format(100*rn_score[1], rn_score[0]))

model = load_model('classifier_vgg16_model.h5')

VGG_score=model.evaluate_generator(test_generator_vgg,verbose=1)

model.summary()

print('Accuracy: {}% \n Loss: {}'.format(100*VGG_score[1], VGG_score[0]))        

resnet_prediction=rn_model.predict(
                                    test_generator_rn,
                                    batch_size=5,
                                    verbose = 1,
                                    )


for i in range(0,5):
  if resnet_prediction[i][0]>resnet_prediction[i][1]:
    print("Result "+str(i+1)+": "+"Negative")
  else:
    print("Result "+str(i+1)+": "+"Positive")


import matplotlib.pyplot as plt
import numpy as np
predicted_samples=test_generator_rn.next()
predicted_images=predicted_samples[0]

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = predicted_images[ind].astype(np.uint8)	# delete .astype(np.uint8) if no imshow error
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('First Six Concrete Images') 
plt.show()