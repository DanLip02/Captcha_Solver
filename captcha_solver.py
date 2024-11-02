import random
import cv2
import numpy as np
import os #for operating system dependent fucntionality
from keras import layers #for building layers of neural net
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import string

class Captchasolver:

    def __init__(self, path_train):
        self.path = path_train
        self.num_captcha = len(os.listdir(path_train))

    def preprocess(self, path, all_captchas):
        character = string.ascii_uppercase + string.digits
        nchar = len(character)
        X = np.zeros((all_captchas, 80, 280, 1))
        y = np.zeros((5, all_captchas, nchar))
        B = sorted(os.listdir(path), key=lambda A: random.random())
        for i, pic in enumerate(B):
            img = cv2.imread(os.path.join(path, pic),cv2.IMREAD_GRAYSCALE)
            if pic.endswith('.png'):
                pic_target = pic[:-4]
                if pic_target.endswith('.png'):
                    pic_target = pic[:-8]

                if len(pic_target) < 6:  # captcha is not more than 5 letters
                    img = img / 255.0  # scales the image between 0 and 1
                    img = np.reshape(img, (80, 280, 1))  # reshapes image to width 200 , height 80 ,channel 1
                    target = np.zeros((5, nchar))  # creates an array of size 5*36 with all entries 0

                    for j, k in enumerate(pic_target):
                        index = character.find(k)
                        target[j, index] = 1  # replaces 0 with 1 in the target array at the position of the letter in captcha

                    X[i] = img  # stores all the images
                    y[:, i] = target  # stores all the info about the letters in captcha of all images
        return X, y

    def createmodel(self, imgshape):
        character = string.ascii_uppercase + string.digits
        nchar = len(character)
        img = layers.Input(shape=imgshape)  # Get image as an input of size 50,200,1
        conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(img)
        bn1 = layers.BatchNormalization()(conv1)
        mp1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bn1)
        conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
        bn2 = layers.BatchNormalization()(conv2)
        mp2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bn2)
        bn2_2 = layers.BatchNormalization()(mp2)
        conv3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(bn2_2)
        bn3 = layers.BatchNormalization()(conv3)  # to improve the stability of model
        mp3 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bn3)
        bn3_3 = layers.BatchNormalization()(mp3)  # t
        conv4 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(bn3_3)
        bn4 = layers.BatchNormalization()(conv4)
        mp4 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bn4)
        bn4_4 = layers.BatchNormalization()(mp4)
        conv5 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(bn4_4)
        bn5 = layers.BatchNormalization()(conv5)
        mp5 = layers.MaxPooling2D((2, 2), padding='same')(bn5)
        flat = layers.Flatten()(mp5)  # convert the layer into 1-D

        outs = []
        for _ in range(5):  # for 5 letters of captcha
            dens = layers.Dense(512, activation='relu')(flat)
            bn = layers.BatchNormalization()(dens)
            # 1 - sigmoid
            # 2 - softmax
            res = layers.Dense(nchar, activation='softmax')(bn)
            outs.append(res)  # result of layers

        # Compile model and return it
        model = Model(img, outs)  # create model
        # optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"] * 5)
        return model

    def fit_model(self, model, x_data, y_data):
        x_train = x_data
        y_train  = y_data
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-5)
        hist = model.fit(x_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], epochs=65,
                         batch_size=60,
                         validation_split=0.2, shuffle=True, callbacks=[reduce_lr, early_stopping])

        preds = model.evaluate(x_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]])
        return hist, preds

    def predict(self, filepath):
        character = string.ascii_uppercase + string.digits
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if img is not None: #image foud at file path
            img = img / 255.0 #Scale image
        res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis])) #np.newaxis=1
        #added this bcoz x_train 970*50*200*1
        #returns array of size 1*5*36
        result = np.reshape(res, (5, 36)) #reshape the array
        k_ind = []
        probs = []
        for i in result:
            k_ind.append(np.argmax(i)) #adds the index of the char found in captcha

        capt = '' #string to store predicted captcha
        for k in k_ind:
            capt += character[k] #finds the char corresponding to the index
        return capt

if __name__ == '__main__':
    train_path = 'train'
    test_path = 'test'
    flag = True
    solver = Captchasolver(train_path)
    path = solver.path
    all_captchas = solver.num_captcha
    model = solver.createmodel(imgshape=(80,280,1))
    model.summary()
    X, y = solver.preprocess(path=path, all_captchas=all_captchas)
    X_train, y_train = X[:9000], y[:, :9000]
    X_test, y_test = X[9000:], y[:, 9000:]
    if flag:
        hist_train, preds_train = solver.fit_model(model, X_train, y_train)
    counter = 0
    for i, pic in enumerate(os.listdir(test_path)):
        pic_target_val_ = pic[:-4]
        if pic_target_val_.endswith('.png'):
            pic_target_val_ = pic[:-8]
        path_pred = test_path + f'/{pic_target_val_}.png'
        predict_c_ = solver.predict(filepath=path_pred)
        print(f'Fact is: {pic_target_val_}')
        print('Predict is: ', predict_c_)
        if predict_c_ == pic_target_val_:
            counter += 1
    print('Correct recognised_CB: ', counter)

