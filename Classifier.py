import numpy as np 
import pandas as pd 
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

with ZipFile('../input/dogs-vs-cats/train.zip') as zipObj:
    zipObj.extractall('../kaggle/working/temp')

filenames = os.listdir('../kaggle/working/temp/train')
categories =[]

for file in filenames:
    category = file.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    elif category == 'cat':
        categories.append('cat')

df = pd.DataFrame({'filename':filenames , 'category':categories})

df['category'].value_counts().plot.bar()

train_data , val_data = train_test_split(df, test_size=0.2, 
                                         random_state=42)

train_data['category'].value_counts().plot.bar()
val_data['category'].value_counts().plot.bar()

model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPool2D(2,2),
    layers.Dropout(0.2),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPool2D(2,2),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D(2,2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy',
             metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=40,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

train_gen = train_datagen.flow_from_dataframe(train_data,
                                              '../kaggle/working/temp/train',
                                             x_col='filename',
                                             y_col='category',
                                             class_mode='binary',
                                             batch_size=32,
                                             target_size=(150,150))


val_datagen = ImageDataGenerator(rescale=1/255)


val_gen = val_datagen.flow_from_dataframe(val_data,
                                          '../kaggle/working/temp/train',
                                             x_col='filename',
                                             y_col='category',
                                             class_mode='binary',
                                             batch_size=32,
                                             target_size=(150,150))


early = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                      verbose=1)
check = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', 
                        verbose=1, save_best_only=True)


history = model.fit_generator(train_gen,
                             validation_data=val_gen,
                             epochs=100, 
                             steps_per_epoch=625,
                             validation_steps=150,
                             callbacks=[early,check],
                             verbose=1)

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'r', label='Training')
plt.plot(epochs, val_acc, 'b', label='Validation')
plt.title('Training Accuracy vs Validation Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training')
plt.plot(epochs, val_loss, 'b', label='Validation')
plt.title('Training Loss vs Validation Loss')
plt.legend()

plt.show()


#Test Dataset
with ZipFile('../input/dogs-vs-cats/test1.zip') as zipObj:
    zipObj.extractall('../kaggle/working/temp')

filenames = os.listdir('../kaggle/working/temp/test1')
test_data = pd.DataFrame({'filename': filenames}) 

train_gen.class_indices

best_model = load_model('best_model.h5')

test_datagen = ImageDataGenerator(rescale=1/255)

test_gen = test_datagen.flow_from_dataframe(test_data,
                                           '../kaggle/working/temp/test1',
                                           x_col='filename',
                                           y_col=None,
                                           class_mode=None,
                                           batch_size=128,
                                           target_size=(150,150))

predict = best_model.predict_generator(test_gen)
final_prediction = np.argmax(predict, axis=1)

predict_df = pd.DataFrame(final_prediction, columns=['label'])
submission_df = test_data.copy()
submission_df['id'] = (submission_df['filename'].str.split('.').str[0]).astype(int)
submission_df = pd.concat([submission_df, predict_df], axis=1)
submission_df = submission_df.drop(['filename'], axis=1)
submission_df = submission_df.sort_values(by=['id'])
submission_df = submission_df.reset_index(drop=True)
submission_df.to_csv('submission.csv', index=False)
