import torch

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 50
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 5

head_size = 16
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8
top_k = 2
vocab_size = 1024



num_heads = n_head
embed_dim = n_embed


import torch

# batch_size = 16 # how many independent sequences will we process in parallel?
# block_size = 32 # what is the maximum context length for predictions?
# max_iters = 2
# eval_interval = 100
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 5

# head_size = 16
# n_embed = 128
# n_head = 8
# n_layer = 8
# dropout = 0.1
# num_experts = 8
# top_k = 2
# vocab_size = 1024



# num_heads = n_head
# embed_dim = n_embed