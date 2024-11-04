from model import *
from parameters import *

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')


    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # print(f"step {iter} ")

    # best_val_loss = losses['val']



    # if losses['val'] < best_val_loss :
    #         best_val_loss = losses['val']
    #         if iter > 0:
    #             checkpoint = {
    #                 # 'model': raw_model.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'model_args': model_args,
    #                 'iter': iter,
    #                 'best_val_loss': best_val_loss,
    #                 'config': config,
    #             }
    #             print(f"saving checkpoint to {out_dir}")
    #             torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

# # %%
# checkpoint = {
#                     # 'model': raw_model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'model_args': model_args,
#                     'iter': iter,
#                     'best_val_loss': best_val_loss,
#                     'config': config,
#                 }


torch.save( model.state_dict(), os.path.join(out_dir, 'ckp0t.pt'))

