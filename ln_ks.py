
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

train_sample = []
train_label = []

for i in range(1000):
    younger_ages = randint(13, 64)
    train_sample.append(younger_ages)
    train_label.append(0)

    older_ages = randint(65, 100)
    train_sample.append(older_ages)
    train_label.append(1)

train_sample = np.array(train_sample)
train_label = np.array(train_label)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_train_sample = scaler.fit_transform(train_sample.reshape(-1, 1))

model = Sequential([Dense(16, input_dim=1, activation='relu'), Dense(32, activation='relu'), Dense(2, activation='softmax')])

model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(scaler_train_sample, train_label, batch_size=10, epochs=10)

test_sample = []
test_label = []

for i in range(500):
    younger_ages = randint(13, 64)
    test_sample.append(younger_ages)
    test_label.append(0)

    older_ages = randint(65, 100)
    test_sample.append(older_ages)
    test_label.append(1)

test_sample = np.array(test_sample)
test_label = np.array(test_label)

# Predict using the same format as used during training
scaler_test_sample = scaler.transform(test_sample.reshape(-1, 1))
predict = model.predict(scaler_test_sample, batch_size=10)

# Convert probabilities to class labels
predict_labels = np.argmax(predict, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(test_label, predict_labels)
print(conf_matrix)








# import keras
# from keras.models import Sequential

# from keras.layers import Dense
# from keras.optimizers import Adam

# import numpy as np
# from random import randint

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix


# train_sample = []
# train_label = []

# for i in range(1000):
#     younger_ages = randint(13, 64)
#     train_sample.append(younger_ages)
#     train_label.append(0)

#     older_ages = randint(65, 100)
#     train_sample.append(older_ages)
#     train_label.append(1)

    
# train_sample = np.array(train_sample)
# train_label = np.array(train_label)

# scaler = MinMaxScaler(feature_range=(0, 1))

# scaler_train_sample = scaler.fit_transform(train_sample.reshape(-1, 1))

# model = Sequential([Dense(16, input_dim=1, activation='relu'), Dense(32, activation='relu'), Dense(2, activation='softmax')])

# model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_sample, train_label, batch_size=10, epochs=10)


# test_sample = []
# test_label = []

# for i in range(500):
#     younger_ages = randint(13, 64)
#     test_sample.append(younger_ages)
#     test_label.append(0)

#     older_ages = randint(65, 100)
#     test_sample.append(older_ages)
#     test_label.append(1)

# test_sample = np.array(test_sample)
# test_label = np.array(test_label)

# predict = model.predict(test_sample, batch_size=10)

# predict_values = confusion_matrix(test_label, predict)
# print(predict_values)

