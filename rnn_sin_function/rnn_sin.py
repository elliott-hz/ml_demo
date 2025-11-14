import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# ======================================
# 1. Generate sine wave data
# ======================================
x = np.linspace(0, 100, 500)        # 500 points from 0 to 100
y = np.sin(x)                       # Sine wave

# Construct training samples: use previous 20 steps to predict the 21st step
TIME_STEPS = 20
X, Y = [], []
for i in range(len(y) - TIME_STEPS):
    X.append(y[i:i+TIME_STEPS])     # First 20 points as input
    Y.append(y[i+TIME_STEPS])       # 21st point as label

X = np.array(X)
Y = np.array(Y)

# Reshape to (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ======================================
# 2. Define model
# ======================================
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(TIME_STEPS, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ======================================
# 3. Train model
# ======================================
model.fit(X, Y, epochs=20, batch_size=16, verbose=1)

# ======================================
# 4. Prediction
# ======================================
pred = model.predict(X, verbose=0)

# ======================================
# 5. Visualization comparison
# ======================================
plt.figure(figsize=(8,4))
plt.plot(y[TIME_STEPS:], label='True Sine')
plt.plot(pred, label='RNN Prediction')
plt.legend()
plt.title("RNN learns to predict sine wave")
plt.show()