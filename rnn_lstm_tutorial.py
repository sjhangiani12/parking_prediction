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
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LENGTH = 12
FUTURE_PERIOD_PREDICT = 1
GARAGE_TO_PREDICT = "SALLING"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LENGTH}--SEQ--{FUTURE_PERIOD_PREDICT}--PRED-{int(time.time())}"


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

