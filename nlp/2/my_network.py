import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        # Initialize LSTM
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        # Initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        # src = [src sent len, batch size]
        
        # Compute embeddings and apply dropout
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim]
        
        # Pass embeddings through RNN
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src sent len, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        # Return outputs, hidden state, and cell state
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        # Initialize LSTM
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        # Initialize fully connected layer
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        
        # Initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        # Add an extra dimension to input
        input = input.unsqueeze(0)
        # input = [1, batch size]
        
        # Compute embeddings and apply dropout
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        
        # Pass embeddings through RNN
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch size, hid dim]
        
        # Remove the extra dimension from output
        prediction = self.out(output.squeeze(0))
        # prediction = [batch size, output dim]
        
        # Return prediction, hidden state, and cell state
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # Encoder outputs
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to the decoder is the <sos> tokens
        input = trg[0, :]
        
        for t in range(1, max_len):
            # Pass through decoder
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Store prediction
            outputs[t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs
