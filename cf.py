import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple neural network model for collaborative filtering
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_games, embedding_dim=10):
        super(RecommenderNet, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.game_embeddings = nn.Embedding(num_games, embedding_dim)

    def forward(self, user, game):
        user_emb = self.user_embeddings(user)
        game_emb = self.game_embeddings(game)
        return (user_emb * game_emb).sum(1)  # Dot product of embeddings


# Initialize the model
model = RecommenderNet(num_users=1000, num_games=5000)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(100):
    optimizer.zero_grad()
    prediction = model(user_tensor, game_tensor)
    loss = loss_fn(prediction, rating_tensor)
    loss.backward()
    optimizer.step()
