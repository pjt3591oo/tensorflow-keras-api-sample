import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model = Sequential()

# feature 인풋 3개 아웃풋 8개
# input_dim으로 피처수 설정
model.add(Dense(units=8, input_dim=3, activation='relu'))
# 인풋 8개, 아웃풋 8개
model.add(Dense(8, activation='relu'))
# 인풋 8개, 아웃풋 3개
model.add(Dense(3, activation='softmax'))

# 손실함수와 최적화 함수 설정
model.compile(loss='mse', optimizer='sgd')

# 학습
model.fit(x, y, epochs=10, verbose=1)

# 예측
pred = model.predict(x)
print(pred)

model.summary()
