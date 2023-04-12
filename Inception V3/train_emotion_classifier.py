"""
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split

# parameters
batch = 32
epoch = 10000
shapee = (48, 48, 1)
split_ratio = .2
verbose = 1
class_no = 7
patience = 50
path = 'xception_models/'

# data generator
datagen = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
xception_model = mini_XCEPTION(shapee, class_no)
xception_model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
xception_model.summary()





    # callbacks
log_file_path = path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
trained_xception_models_path = path + '_mini_XCEPTION'
xception_model_names = trained_xception_models_path + '.hdf5'
xception_model_checkpoint = xception_modelCheckpoint(xception_model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
callbacks = [xception_model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, class_no = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
xception_model.fit_generator(datagen.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch,
                        epochs=epoch, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
