# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Generate synthetic time series data
# np.random.seed(0)
# time = np.arange(0, 100, 0.1)
# data = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

# # Introduce anomalies
# actual_anomalies = np.random.choice(len(time), size=10, replace=False)
# data[actual_anomalies] += np.random.normal(scale=5, size=len(actual_anomalies))

# # Plot the data
# plt.plot(time, data)
# plt.scatter(time[actual_anomalies], data[actual_anomalies], color='red')
# plt.show()

# # Prepare the data
# data = data.reshape(-1, 1)
# data = torch.tensor(data, dtype=torch.float32)

# # Define the autoencoder neural network
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(4, 8),
#             nn.ReLU(),
#             nn.Linear(8, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# # Instantiate the model
# model = Autoencoder()

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# epochs = 100
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(data)
#     loss = criterion(outputs, data)
#     loss.backward()
#     optimizer.step()
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # Detect anomalies
# model.eval()
# with torch.no_grad():
#     outputs = model(data)
#     loss = criterion(outputs, data)
#     reconstruction_error = torch.abs(outputs - data).numpy()

# # Set a threshold for anomaly detection
# threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

# # Identify anomalies
# detected_anomalies = np.where(reconstruction_error > threshold)[0]

# # Calculate accuracy
# true_positives = len(set(detected_anomalies).intersection(set(actual_anomalies)))
# false_positives = len(set(detected_anomalies) - set(actual_anomalies))
# false_negatives = len(set(actual_anomalies) - set(detected_anomalies))
# accuracy = true_positives / (true_positives + false_positives + false_negatives)

# print(f"Accuracy of the anomaly detection: {accuracy * 100:.2f}%")

# # Plot the results
# plt.plot(time, data.numpy())
# plt.scatter(time[detected_anomalies], data.numpy()[detected_anomalies], color='red')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic time series data
np.random.seed(0)
time = np.arange(0, 100, 0.1)
data = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

# Introduce anomalies
actual_anomalies = np.random.choice(len(time), size=10, replace=False)
data[actual_anomalies] += np.random.normal(scale=5, size=len(actual_anomalies))

# Plot the data
plt.plot(time, data)
plt.scatter(time[actual_anomalies], data[actual_anomalies], color='red')
plt.show()

# Prepare the data
data = data.reshape(-1, 1)
data = torch.tensor(data, dtype=torch.float32)

# Define the autoencoder neural network
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Detect anomalies
model.eval()
with torch.no_grad():
    outputs = model(data)
    loss = criterion(outputs, data)
    reconstruction_error = torch.abs(outputs - data).numpy()

# Set a threshold for anomaly detection
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

# Identify anomalies
detected_anomalies = np.where(reconstruction_error > threshold)[0]

# Calculate accuracy
true_positives = len(set(detected_anomalies).intersection(set(actual_anomalies)))
false_positives = len(set(detected_anomalies) - set(actual_anomalies))
false_negatives = len(set(actual_anomalies) - set(detected_anomalies))
accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0

print(f"Accuracy of the anomaly detection: {accuracy * 100:.2f}%")

# Plot the results
plt.plot(time, data.numpy())
plt.scatter(time[detected_anomalies], data.numpy()[detected_anomalies], color='red')
plt.show()