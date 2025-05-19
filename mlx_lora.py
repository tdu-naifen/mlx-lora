import argparse
import time
import types
import matplotlib.pyplot as plt
import datetime
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run
import wandb
import numpy as np

world = mx.distributed.init(backend="mpi")
size = world.size()

def all_reduce_grads(grads):
    if size == 1:
        return grads
    return tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)

class ReportingCallback(TrainingCallback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_after_backward(self, model, grads, step):
        # Compute gradient norm
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in tree_map(lambda x: x, grads).values()))
        # Log gradient norm for this rank
        wandb.log({f"train/grad_norm_rank_{world.rank()}": float(grad_norm), "iteration": step})
        return all_reduce_grads(grads)

    def on_train_loss_report(self, info):
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            train_perplexity = float(np.exp(train_loss))
            # Log training metrics for this rank
            print(f"[Train Rank {world.rank()}] Iteration {iteration}: Loss = {train_loss:.4f}, Perplexity = {train_perplexity:.4f}", flush=True)
            wandb.log({
                f"train/loss_rank_{world.rank()}": train_loss,
                f"train/perplexity_rank_{world.rank()}": train_perplexity,
                "iteration": iteration
            })

    def on_val_loss_report(self, info):
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            val_perplexity = float(np.exp(val_loss))
            # Log validation metrics for this rank
            print(f"[Valid Rank {world.rank()}] Iteration {iteration}: Loss = {val_loss:.4f}, Perplexity = {val_perplexity:.4f}", flush=True)
            wandb.log({
                f"val/loss_rank_{world.rank()}": val_loss,
                f"val/perplexity_rank_{world.rank()}": val_perplexity,
                "iteration": iteration
            })

    def on_optimizer_update(self, info):
        iteration = info.get("iteration")
        lr = info.get("learning_rate", None)
        if iteration is not None and lr is not None:
            wandb.log({f"train/learning_rate_rank_{world.rank()}": lr, "iteration": iteration})

def plot_metrics(metrics_callback, save_path=None):
    if not metrics_callback.train_losses and not metrics_callback.val_losses:
        print("No metrics to plot.")
        return
    plt.figure(figsize=(20, 10))
    if metrics_callback.train_losses:
        train_its, train_vals = zip(*metrics_callback.train_losses)
        plt.plot(train_its, train_vals, '-o', label='Train Loss')
    if metrics_callback.val_losses:
        val_its, val_vals = zip(*metrics_callback.val_losses)
        plt.plot(val_its, val_vals, '-o', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        wandb.log({"loss_curve": wandb.Image(save_path)})
    plt.close()

def main():
    if size == 1:
        print("Single process mode: no gradient averaging needed.", flush=True)
    else:
        print(f"Distributed mode: Rank {world.rank()} - averaging gradients across {size} ranks.", flush=True)

    parser = argparse.ArgumentParser(description="Fine-tune LLM as a classifier with MLX LM + LoRA.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="data/")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=25)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="jumbo_adapters")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--testfile", type=str, default="$HOME/project/test.jsonl")
    parser.add_argument("--test-batches", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--lora-parameters", type=dict, default={
        "keys": ['mlp.gate_proj', 'mlp.down_proj', 'self_attn.q_proj', 'mlp.up_proj', 'self_attn.v_proj', 'self_attn.k_proj'],
        "rank": 64,
        "alpha": 64,
        "dropout": 0.2,
        "scale": 16.0
    })
    parser.add_argument("--lr-schedule", type=dict, default={
        "name": "cosine_decay",
        "warmup": 200,
        "warmup_init": 1e-7,
        "arguments": [1e-5, 500, 1e-7]
    })
    args = parser.parse_args()

    start_time = time.time()

    wandb.login(key="")  # Replace with secure key management
    wandb.init(project="medical-diagnosis-classifier", config=vars(args), settings=wandb.Settings(start_method="thread"))

    run_name = f"run-rank-{world.rank()}"
    # Log key hyperparameters explicitly
    wandb.summary.update({
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "iterations": args.iters,
        "model": args.model,
        "name": run_name,
    })

    wandb.summary.update({
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "iterations": args.iters,
        "model": args.model
    })

    metrics_callback = ReportingCallback()

    # Run fine-tuning
    run(types.SimpleNamespace(**vars(args)), training_callback=metrics_callback)

    # Plot metrics
    metrics_name = f"graphs/metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_metrics(metrics_callback, save_path=metrics_name)

    end_time = time.time()
    print(f"Script execution time: {end_time - start_time:.2f} seconds", flush=True)
    wandb.finish()

if __name__ == "__main__":
    main()