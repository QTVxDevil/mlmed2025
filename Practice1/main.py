# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, welch

# %%
mitbih_train = pd.read_csv('./Practice1/dataset/mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('./Practice1/dataset/mitbih_test.csv', header=None)

datasets = {
    "MIT-BIH Train": mitbih_train,
    "MIT-BIH Test": mitbih_test,
}


# %%
for name, df in datasets.items():
    print(f"Dataset: {name}")
    print(df.shape)  
    print(df.head()) 
    print(df.info()) 
    print("--" * 50)


# %%
for name, df in datasets.items():
    print(f"{name} missing values:\n{df.isnull().sum().sum()} values missing\n")


# %%
for name, df in datasets.items():
    print(f"{name} duplicates: {df.duplicated().sum()} duplicate rows")

# %%
for name, df in datasets.items():
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df.iloc[:, -1], palette='coolwarm')
    plt.title(f"Class Distribution in {name}")
    plt.xlabel("Heartbeat Class")
    plt.ylabel("Count")
    plt.show()


# %%
plt.figure(figsize=(14, 8))

for i, (name, df) in enumerate(datasets.items()):
    plt.subplot(2, 2, i + 1)
    plt.plot(df.iloc[0, :-1]) 
    plt.title(f"ECG Signal Sample from {name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()


# %%
for name, df in datasets.items():
    print(f"Statistical Summary of {name}:\n")
    print(df.describe().T)
    print("\n" + "--" * 50 + "\n")


# %%
signal = mitbih_train.iloc[0, :-1].values
peaks, _ = find_peaks(signal, distance=10, prominence=0.2)

plt.figure(figsize=(10, 4))
plt.plot(signal, label="ECG Signal")
plt.plot(peaks, signal[peaks], "rx", label="Detected Peaks")
plt.title("Peak Detection in ECG Signal")
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


# %%
plt.figure(figsize=(12, 6))

for i, (name, df) in enumerate(datasets.items()):
    fs = 125
    f, Pxx = welch(df.iloc[0, :-1], fs=fs, nperseg=128)

    plt.plot(f, Pxx, label=name)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Frequency Analysis of ECG Signals")
plt.legend()
plt.show()


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# %%
np.random.seed(42)
tf.random.set_seed(42)

# %%
print(f"MIT-BIH Train Shape: {mitbih_train.shape}")
print(f"MIT-BIH Test Shape: {mitbih_test.shape}")

# %%
X_train, y_train = mitbih_train.iloc[:, :-1].values, mitbih_train.iloc[:, -1].values
X_test, y_test = mitbih_test.iloc[:, :-1].values, mitbih_test.iloc[:, -1].values

print(f"Training Data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing Data: {X_test.shape}, Labels: {y_test.shape}")


# %%
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Final Training Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
print(f"Final Testing Shape: {X_test.shape}, Labels Shape: {y_test.shape}")


# %%
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    LSTM(64, return_sequences=True),
    LSTM(32),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# %%
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# %%
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Model Loss")

plt.show()


# %%
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true_classes, y_pred_classes))


# %%



