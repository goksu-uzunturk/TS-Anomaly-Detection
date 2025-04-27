import torch
import torch.nn as nn
import numpy as np

class TransformerArchitecture(nn.Module):
    def __init__(self, input_dim, embedding_size, num_heads, num_layers, num_classes, window_size, num_future_samples, dropout=0.1):
        super(TransformerArchitecture, self).__init__()
        """
        Base class for Transformer-based models for anomaly detection.

        Args:
            input_dim (int): Number of features per time step.
            embedding_size (int): Dimensionality of the embedding space.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer encoder/decoder layers.
            num_classes (int): Number of unique tag classes.
            window_size (int): Number of time steps in each patch.
            num_future_samples (int): Number of future tags to predict.
        """

        # Embedding layer for input features
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.Dropout(dropout)
        )

        # Generate positional encoding for the patch size and embedding size
        self.positional_encoding = nn.Parameter(self.generate_positional_encoding(window_size, embedding_size), requires_grad=False)

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dropout=dropout,  batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Learnable queries for the decoder
        self.query = nn.Parameter(torch.zeros(num_future_samples, embedding_size))

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Classification layer for predicting tags
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward_encoder(self, x):
        """Encodes the input sequence."""
        return self.encoder(x)

    def forward_decoder(self, encoder_output):
        """Decodes the output using learnable queries."""
        batch_size = encoder_output.size(0)
        # Generate causal mask for the decoder
        causal_mask = self.generate_causal_mask(self.query.size(0)).to(encoder_output.device)  # Mask shape: (num_future_samples, num_future_samples)
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        
        return self.decoder(query, encoder_output, tgt_mask=causal_mask)
    
    def generate_causal_mask(self, seq_len):
        """
        Generate a causal mask to prevent attending to future time steps.
        
        Args:
            seq_len (int): Length of the input sequence.
        
        Returns:
            torch.Tensor: A causal mask of shape (seq_len, seq_len).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Mask future positions with -inf
        return mask  # Shape: (seq_len, seq_len)
    

    def generate_positional_encoding(self, seq_len, embedding_size):
        """
        Generate sinusoidal positional encoding for the input sequence.

        Args:
            seq_len (int): The length of the input sequence.
            embedding_size (int): The size of the embedding vector.

        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, seq_len, embedding_size).
        """
        position = torch.arange(seq_len).unsqueeze(1).float()  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(np.log(10000.0) / embedding_size))  # (embedding_size // 2)

        # Create the positional encoding matrix
        pe = torch.zeros(seq_len, embedding_size)  # (seq_len, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        return pe.unsqueeze(0)  # Add batch dimension (1, seq_len, embedding_size)
    
    def forward(self, x):
        """
        Forward pass for tag prediction.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_dim).
        Returns:
            torch.Tensor: Predicted tags of shape (batch_size, num_future_samples, num_classes).
        """
        # Embed input features and add positional encoding
        x = self.embedding(x) + self.positional_encoding  # Shape: (batch_size, window_size, embedding_size)

        # Pass through the encoder
        encoder_output = self.forward_encoder(x)  # Shape: (batch_size, window_size, embedding_size)

        # Pass through the decoder
        decoder_output = self.forward_decoder(encoder_output)  # Shape: (batch_size, num_future_samples, embedding_size)

        # Apply the classifier to each time step in num_future_tags
        out = self.classifier(decoder_output)  # Shape: (batch_size, num_future_samples, num_classes)

        return out

    def train_model(self, *args, **kwargs):
        raise NotImplementedError("Train method should be implemented in the derived classes.")


class Predictor(TransformerArchitecture):
    def __init__(self, input_dim, embedding_size, num_heads, num_layers, num_classes, window_size, num_future_samples, dropout):
        super(Predictor, self).__init__(
            input_dim=input_dim, 
            embedding_size=embedding_size, 
            num_heads=num_heads, 
            num_layers=num_layers, 
            num_classes=num_classes, 
            window_size=window_size, 
            num_future_samples=num_future_samples,
            dropout=dropout
        )

    def train_model(self, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, num_epochs, device, patience=5):
        model.to(device)
        train_losses, val_losses = [], []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

                # Record loss
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

            # Scheduler update
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return train_losses, val_losses

