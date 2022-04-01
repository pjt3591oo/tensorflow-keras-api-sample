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

class MyModel(Model):
  
  def __init__(self):
    super().__init__()
    self.d1 = Dense(8, activation='relu')
    self.d2 = Dense(8, activation='relu')
    self.d3 = Dense(3, activation='softmax')

  def call(self, input):
    hidden1 = self.d1(input)
    hidden2 = self.d2(hidden1)
    output = self.d3(hidden2)

    return output

  def summary(self):
    inputs = Input((None, 3))
    Model(inputs, self.call(inputs)).summary()

model = MyModel()
# input_shape: (데이터 수, feature)
# 데이터 수를 None을 집어넣으면 n개의 데이터를 지정하는 것이 가능
model.build(input_shape=(None, 3)) 

# 모델 요약
model.summary()
model.compile(loss='mse', optimizer='sgd')

# 학습
model.fit(x, y, epochs=10, verbose=1)

# 예측
pred = model.predict(x)
print(pred)

model.summary()
