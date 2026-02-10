######################
bnkausik at gmail.com
code for the recurrent tranformers in the paper https://arxiv.org/abs/2402.14746
######################

out_dir = 'out-shakespeare-char'
dataset = 'shakespeare_char'




########my tweaks
compile=False
block_size =  256 #context length
eval_iters=40
eval_interval=250
batch_size = 64
max_iters = 15000
learning_rate = 1e-3
min_lr = learning_rate/10
device="mps"
target_loss = 1.3
decay_lr = False
always_save_checkpoint=False

rec_block = 16 #recurrence block size; if 0: regular transformer
dropout=0.2

# model size
n_layer=1
n_head =6
n_embd=384



