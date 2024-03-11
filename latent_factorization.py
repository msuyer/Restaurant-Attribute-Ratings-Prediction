import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define hyperparameters
LEARNING_RATE = [0.01, 0.001]
BATCH_SIZE = [256, 512]
NUM_EPOCHS = [25, 50]
EMBEDDING_DIM = [16, 32]

# Load data from CSV file
data = pd.read_csv('Reviews_Part1.csv', usecols=['user_id', 'business_id', 'stars'])

# Convert user_id and business_id to integers
data["user_id"] = pd.factorize(data["user_id"])[0]
data["business_id"] = pd.factorize(data["business_id"])[0]

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)


# Define PyTorch dataset for loading data
class RatingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["user_id"], row["business_id"], row["stars"]


# Define latent factorization model using PyTorch
class LatentFactorization(torch.nn.Module):
    def __init__(self, num_users, num_businesses, embedding_dim):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.business_embedding = torch.nn.Embedding(num_businesses, embedding_dim)
        self.user_bias = torch.nn.Embedding(num_users, 1)
        self.business_bias = torch.nn.Embedding(num_businesses, 1)

    def forward(self, user_ids, business_ids):
        user_embedded = self.user_embedding(user_ids)
        business_embedded = self.business_embedding(business_ids)
        user_bias = self.user_bias(user_ids)
        business_bias = self.business_bias(business_ids)
        dot_product = torch.sum(user_embedded * business_embedded, dim=1)
        predictions = dot_product + user_bias.squeeze() + business_bias.squeeze()

        return predictions


# Initialize model
num_users = len(data["user_id"].unique())
num_businesses = len(data["business_id"].unique())

# Define loss function and optimizer
criterion = torch.nn.MSELoss()

# Initialize best hyperparameter values
best_loss = 9999
best_lr = 0
best_b = 0
best_e = 0
best_n = 0

# Iterate through hyperparameter values
for lr in LEARNING_RATE:
    for b in BATCH_SIZE:
        for e in EMBEDDING_DIM:
            for n in NUM_EPOCHS:

                model = LatentFactorization(num_users, num_businesses, e)

                # Move model and data to GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Define PyTorch dataloaders for loading data
                batch_size = b
                train_loader = DataLoader(RatingDataset(train_data), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(RatingDataset(val_data), batch_size=batch_size, shuffle=True)

                # Train model
                num_epochs = n
                for epoch in range(num_epochs):
                    train_loss = 0
                    model.train()
                    for user_ids, business_ids, stars in train_loader:
                        user_ids = user_ids.to(device)
                        business_ids = business_ids.to(device)
                        stars = stars.to(device)
                        optimizer.zero_grad()
                        predictions = model(user_ids, business_ids)
                        loss = criterion(predictions, stars.float())
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * len(user_ids)
                    train_loss /= len(train_data)
                    print('Epoch ' + str(epoch + 1) + ', Loss: ' + str(train_loss))

                # Validate model
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for user_ids, business_ids, stars in val_loader:
                        user_ids = user_ids.to(device)
                        business_ids = business_ids.to(device)
                        stars = stars.to(device)
                        predictions = model(user_ids, business_ids)
                        loss = criterion(predictions, stars.float())
                        val_loss += loss.item() * len(user_ids)
                    val_loss /= len(val_data)

                    # Update best hyperparameters
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_e = e
                        best_b = b
                        best_lr = lr
                        best_n = n
                    print('Validation Loss: ' + str(val_loss))

# Print best loss and best epochs
print('best loss: ' + str(best_loss) + ' best num epochs: ' + str(best_n) + ' best batch size: ' + str(
    best_b) + ' best embedding dim: ' + str(best_e) + ' best learning rate: ' + str(best_lr))