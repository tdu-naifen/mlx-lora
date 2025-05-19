# MLX distributed LORA

This is a wild west script, and the purpose is to do some very small scale LoRA over multiple Macs.

Doc: https://medium.com/@dutingzhen/distributed-lora-through-mlx-035c48597848

Eventuall can be run in parallel with following command: 

```
mpirun --hostfile hostfile -np 2 \
--mca oob_tcp_if_include bridge0 \
--mca btl_tcp_if_include bridge0 \
--verbose \
$HOME/miniconda3/condabin/conda run -n mlx_finetuning python $HOME/PycharmProjects/finetuning/mlx_lora.py
```
