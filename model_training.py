from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

X_train = np.random.rand(100, 20)
y_train = np.random.randint(2, size=100)

model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16)

model.save("gesture_model.h5")
