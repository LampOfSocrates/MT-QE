#/user/HS401/ss05548/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=131.227.65.142 --master_port=1234 /user/HS401/ss05548/Code/MT-QE/src/train_model2.py
#fabric run model --accelerator="gpu" --devices=1 --num_nodes=2 --node_rank=0 --main_address="131.227.65.142" --main-port=1234 /user/HS401/ss05548/Code/MT-QE/src/train_model2.py
torchrun --nproc_per_node=1 --nnodes=2 --node-rank=1 --master-addr="131.227.65.142" --master-port=1234 /user/HS401/ss05548/Code/MT-QE/src/train_model2.py
