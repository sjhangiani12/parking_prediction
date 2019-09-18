#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LENGTH = 12
FUTURE_PERIOD_PREDICT = 1
GARAGE_TO_PREDICT = "SALLING"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LENGTH}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}.h5"


df = pd.read_csv("data/cleaned_df.csv")
df = pd.pivot_table(
    df,
    values="percentage",
    index=["date"],
    columns=["garageCode"],
    aggfunc=np.mean,
    fill_value=0,
).reset_index()


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


df["future"] = df[f"{GARAGE_TO_PREDICT}"].shift(-FUTURE_PERIOD_PREDICT)
df["target"] = list(map(classify, df[f"{GARAGE_TO_PREDICT}"], df["future"]))
df.set_index("date", inplace=True)

times = sorted(df.index.values)
last_5pct = times[-int(0.05 * len(times))]

validation_main_df = df[(df.index >= last_5pct)]
main_df = df[(df.index < last_5pct)]


def preprocess_df(df):
    df = df.drop("future", 1)
    # skip the scaling sicne we're working w percentages and its already scaled correctly
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LENGTH)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LENGTH:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    increases = []
    decreases = []

    for seq, target in sequential_data:
        if target == 0:
            decreases.append([seq, target])
        elif target == 1:
            increases.append([seq, target])

    random.shuffle(increases)
    random.shuffle(decreases)

    lower = min(len(increases), len(decreases))

    increases = increases[:lower]
    decreases = decreases[:lower]

    sequential_data = increases + decreases
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Decreases: {train_y.count(0)}, increases: {train_y.count(1)}")
print(
    f"Validation decreases: {validation_y.count(0)}, increases: {validation_y.count(1)}"
)

model = Sequential()
model.add(LSTM(128, input_shape=(
    train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# unique file name that will include the epoch and the validation acc for that epoch
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc',
                                                      verbose=1, save_best_only=True, mode='max'))  # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))
print("Saved model to disk")
