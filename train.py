import time

from model import * 
from config import *
from data import *
from setting import *

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


for iter in range(max_iters):

    # sample a batch of data
    # xb, yb = get_batch('train', batch_size)
    x, y = get_batch(
    split='train', 
    batch_size=batch_size, 
    max_seq_len=config.max_position_embeddings, 
    device=config.device
)
    # train
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, batch_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")


torch.save(model.state_dict(), '{max_iters}.pth')
