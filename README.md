# Experiment 9: Neural Network (Keras) on Non-Linear Toy Data (make_moons)

## AIM
To train a small feedforward neural network using Keras to classify non-linear data generated using the `make_moons` dataset.

## ALGORITHM
1. Import the required Python libraries.
2. Generate a non-linear dataset using `make_moons`.
3. Split the dataset into training and testing sets using an 80:20 ratio.
4. Create a Sequential Neural Network model.
5. Add a hidden Dense layer with 8 neurons and ReLU activation.
6. Add an output Dense layer with 1 neuron and Sigmoid activation.
7. Compile the model using the Adam optimizer and Binary Crossentropy loss function.
8. Train the model for 30 epochs with a batch size of 16.
9. Predict the test dataset values.
10. Calculate and display the classification accuracy.

## THEORY
Neural Networks are machine learning models inspired by the structure of the human brain. They consist of interconnected layers of neurons that learn patterns from input data.

Non-linear activation functions such as ReLU and Sigmoid help neural networks learn complex patterns and decision boundaries. According to the Universal Approximation Theorem, a feedforward neural network with a suitable architecture and non-linear activation can approximate a wide range of continuous functions.

The `make_moons` dataset is a synthetic non-linear dataset containing two interleaving moon-shaped classes. It is used to demonstrate the ability of neural networks to classify non-linear data.

## REQUIREMENTS
- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- NumPy

## INPUT (Python Program)

```python
# EXPERIMENT 9: Neural Network (Keras) on Non-Linear Toy Data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate non-linear dataset
x, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# Split dataset into training and testing sets
xtr, xte, ytr, yte = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create Neural Network model
model = Sequential([
    Input(shape=(2,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    xtr, ytr,
    epochs=30,
    batch_size=16,
    verbose=0
)

# Predict test data
y_pred = (model.predict(xte, verbose=0) > 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = accuracy_score(yte, y_pred)

# Display output
print('NN test accuracy:', round(accuracy, 4))
```

## OUTPUT

```text
NN test accuracy: 0.8700
```

## RESULT

Thus, a feedforward Neural Network using Keras was successfully implemented on the non-linear `make_moons` dataset. The model achieved a test accuracy of **87%**. The experiment demonstrates how neural networks with ReLU and Sigmoid activation functions can learn non-linear patterns and classify complex datasets.

## CONCLUSION
The experiment demonstrates that neural networks can effectively classify non-linear datasets by learning complex decision boundaries through hidden layers and non-linear activation functions.
