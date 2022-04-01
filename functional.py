import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

x = np.array([
  [73., 80., 75.],
  [93., 88., 93.],
  [89., 91., 90.],
  [96., 98., 100.],
  [73., 66., 70.] 
])

y = np.array([
  [152.], [185.], [180.], [196.], [142.]
])

inputs = Input(shape=(3, )) # 3개의 입력층(피처수)을 받는다.
hidden1 = Dense(8, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
outputs = Dense(3, activation='softmax')(hidden2)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='mse', optimizer='sgd')

# 학습
model.fit(x, y, epochs=10, verbose=1)

# 예측
pred = model.predict(x)
print(pred)

model.summary()


