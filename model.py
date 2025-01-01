import torch
import torch.nn as nn
import torch.nn.functional as F

from data import *
from moe import *
from config import *

class blues(nn.Module):
    

    def __init__(self, config, tokenizer):

        super().__init__()
        self.config = config

        # Ensure hidden_size is cleanly divisible by num_attention_heads for proper splitting and combining
        assert config.hidden_size % config.num_attention_heads == 0

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer

        # Embedding matrix for converting tokens to the initial residual state and logits
        self.embedder = nn.Embedding(self.vocab_size, config.hidden_size)

        # Initialize DecoderLayers based on the number specified in config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])

        # Final RMS normalization layer to stabilize the output
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # CrossEntropyLoss criterion for training
        self.criterion = nn.CrossEntropyLoss()

        # Weighting parameter for the MoE loss
        self.lambadada = config.lambadada

    def calc_moe_loss(self, routing_probs_list):
        """
        Calculates the Mixture of Experts (MoE) loss based on routing probabilities.

        Args:
            routing_probs_list (list): List of routing probabilities from each layer.

        Returns:
            torch.Tensor: Cumulative variance-based MoE loss.
        """
        # Concatenate routing probabilities along a new dimension
        all_routing_probs = torch.cat([x.unsqueeze(0) for x in routing_probs_list], dim=0)

        # Calculate expert usage across batch and sequence dimensions
        expert_usage = all_routing_probs.sum(dim=(1, 2))

        # Calculate mean and variance across experts and layers
        usage_mean = expert_usage.mean(dim=0)
        expert_variance = ((expert_usage - usage_mean) ** 2).mean(dim=0)

        # Sum variance across experts
        cum_var = expert_variance.sum()

        return cum_var

    def forward(
        self,
        input_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor = None,
    ) -> torch.Tensor:
       
        training = False if target_token_ids is None else True

        # Convert input tokens to initial residual state using the embedding matrix
        x = self.embedder(input_token_ids) * self.config.hidden_size ** 0.5  # Grok normalizes embedding by sqrt(hidden_size)

        routing_probs_list = []  # List to store routing probabilities of each layer
        # Process input through each DecoderLayer
        for layer in self.layers:
            x, routing_probs = layer(x, training)
            if training:
                routing_probs_list.append(routing_probs)

        # Apply final normalization to the output of the last DecoderLayer
        x = self.final_norm(x)

        # Get the weights of the embedding matrix for use as the output layer
        embedder_weight = self.embedder.weight

        # Calculate logits by matrix multiplication of final output and embedding weights
        logits = torch.matmul(x, embedder_weight.t())

        if training:
            batch_size, input_len, vocab_size = logits.shape

            # Flatten logits and target_token_ids for CrossEntropyLoss
            CEloss = self.criterion(logits.view(batch_size * input_len, vocab_size),
                                    target_token_ids.view(batch_size * input_len))

            # Calculate MoE loss to encourage usage of all experts
            MoEloss = self.calc_moe_loss(routing_probs_list)

            # Combined loss with weighting parameter lambda
            loss = CEloss + MoEloss * self.lambadada
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def Sampler(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Generates token predictions from logits using sampling techniques.

        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, vocab_size).
            temperature (float): Temperature scaling factor for softmax.
            top_p (float): Top-p (nucleus) sampling threshold.
            top_k (int): Top-k sampling threshold.

        Returns:
            torch.Tensor: Predicted token indices of shape (batch_size,).
        """
        logits = logits[:, -1, :]  # Select logits for the last token

        # Apply temperature scaling to logits
        logits.div_(temperature)

        # Calculate softmax probabilities
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # Sort probabilities for top-p and top-k sampling
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p sampling mask
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_p
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        # Apply top-k sampling mask
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1) >= top_k
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalize probabilities
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # Rearrange probabilities back to original order
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        # Sample from the probability distribution
        next_token_id = torch.multinomial(probs, num_samples=1)

        return next_token_id

    def generate(
        self,
        prompt: str,
        output_len: int = 100,
        temperature: float = 0.95,
        top_p: float = 1.0,
        top_k: int = 65,
    ) -> str:
        """
        Generates text based on a prompt using the minGrok model.

        Args:
            prompt (str): Input prompt for text generation.
            output_len (int): Length of the generated text in tokens.
            temperature (float): Temperature scaling factor for sampling.
            top_p (float): Top-p (nucleus) sampling threshold.
            top_k (int): Top-k sampling threshold.

        Returns:
            str: Generated text based on the prompt.
        """
        tokens = self.tokenizer.encode(prompt)  # Encode input prompt into token indices
        tokens = torch.tensor(tokens, device=self.config.device).unsqueeze(0)  # Convert to tensor

        # Check that generated output length does not exceed maximum allowed sequence length
        assert len(tokens) + output_len <= self.config.max_position_embeddings

        for _ in range(output_len):
            logits, _ = self(tokens[:, :self.max_seq_len])  # Get logits from model

            next_token = self.Sampler(
                logits=logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            # Append predicted token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        # Decode token indices to text
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output




class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, use_scale=True):
        """
        Initialize RMSNorm module.

        Args:
            num_features (int): Number of input features.
            eps (float): Small value added to the denominator for numerical stability.
            use_scale (bool): Whether to use learnable scale parameter.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features)) if use_scale else None

    def forward(self, x):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Normalized and optionally scaled tensor.
        """
        # Calculate the mean squared value for each feature
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize inputs
        x = x * torch.rsqrt(mean_squared + self.eps)

        # Apply scale if it exists
        if self.scale is not None:
            x = x * self.scale

        return x
    



class DecoderLayer(nn.Module):
    """
    A decoder layer that integrates the Attention mechanism and MoE. It includes
    normalization steps both before and after the MQA and MoE but never actually normalizes the residual connection.
    """

    def __init__(self, config):
        """
        Initializes a DecoderLayer.

        Args:
            config (object): Configuration object containing model parameters.
        """
        super().__init__()

        # Multi-Query Attention (MQA) module
        self.mqa = MQA(config)

        # Mixture of Experts (MoE) layer
        self.moe = MoELayer(
            model_dim=config.hidden_size,
            expert_hidden_dim=config.hidden_size * config.embedding_multiplier_scale,
            tot_num_experts=config.tot_num_experts,
            chosen_num_experts=config.chosen_num_experts,
            noise_std=config.noise_std
        )

        # RMS normalization layers
        self.pre_mqa_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.post_mqa_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)

        # Dropout layer
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass of the DecoderLayer.

        Args:
            x (torch.Tensor): Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            torch.Tensor: Output tensor after processing through MQA and MoE.
            torch.Tensor: Routing probabilities from the MoE layer (only returned during training).
        """
        # Apply MQA with normalization before and after
        if training:
            x = x + self.drop(self.post_mqa_norm(self.mqa(self.pre_mqa_norm(x))))
            moe_out, routing_probs = self.moe(self.pre_moe_norm(x), training)
            x = x + self.drop(self.post_moe_norm(moe_out))
        else:
            x = x + self.post_mqa_norm(self.mqa(self.pre_mqa_norm(x)))
            moe_out, routing_probs = self.moe(self.pre_moe_norm(x), training)
            x = x + self.post_moe_norm(moe_out)

        return x, routing_probs


chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()_-+={}[]:;\"'<>,./?")


# from typing import List

# class CharacterTokenizer:
#     def __init__(self, chars: List[str]):
#         # Create a dictionary mapping characters to indices
#         self.stoi = {ch: i for i, ch in enumerate(chars)}

#         # Create a dictionary mapping indices back to characters
#         self.itos = {i: ch for i, ch in enumerate(chars)}

#     def encode(self, s: str) -> List[int]:
#         # Use dictionary 'stoi' to convert each character in 's' to its corresponding ID
#         return [self.stoi.get(c, 0) for c in s]

#     def decode(self, t: List[int]) -> str:
#         # Use dictionary 'itos' to convert each ID in 't' back to its corresponding character
#         return ''.join([self.itos.get(i, '') or '' for i in t])


# tokenizer = CharacterTokenizer(chars)




from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pandas as pd
from data import dataset
from data import batch_size

# Initialize a Byte Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer for building vocab
trainer = trainers.BpeTrainer(vocab_size=30_000, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])

# Initialize a set to store unique characters
unique_chars = set()

# Load CSV file and prepare text for training the tokenizer
texts = []
for chunk in pd.read_csv(dataset, chunksize=batch_size):
    # Extract text and add to the list for tokenizer training
    texts.extend(chunk["Text"].tolist())
    # Update unique characters set
    for text in chunk["Text"]:
        unique_chars.update(text)

# Print unique characters
print("Unique characters:", unique_chars)
print("Number of unique characters:", len(unique_chars))

# Train tokenizer on the text data
tokenizer.train_from_iterator(texts, trainer)

# Get vocab size
vocab_size = tokenizer.get_vocab_size()
print("Subword-level vocab size:", vocab_size)




@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters = 10): # to periodically estimate loss during the training loop
    out = {}
    model.eval() # sets model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # just resets to training mode
    return out





model = blues(config, tokenizer).to(config.device)
