#!/usr/bin/env python3
"""Runs the train script with a grid of hyperparameters."""

import os
import subprocess
from multiprocessing import Process, Queue
import torch

GPUS = [0,1,2,3]
DRY_RUN = False  # Set to False to actually run the experiments

experiment_configs = ["configs/FTW-25-Experiments-1-1/experiment-1-1.1", "configs/FTW-25-Experiments-1-1/experiment-1-1.2", "configs/FTW-25-Experiments-1-1/experiment-1-1.3", "configs/FTW-25-Experiments-1-1/experiment-1-1.4"]

def run_experiments(work: "Queue[str]", gpu_idx: int) -> None:
    """Run experiments from the queue."""
    print(f"Running {work.qsize()} experiments")
    print(f"work.empty(): {work.empty()}")
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(f"Running: {experiment}")
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))

if __name__ == "__main__":
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print('Available GPUs: ', num_gpus)

    work: "Queue[str]" = Queue()

    # Add the experiments to the queue with the GPU index
    for config in experiment_configs:
        command = (
              f"ftw model fit"
            + f" --config {config}.yaml"
            + f" --trainer.devices [GPU]"
        )
        work.put(command)

    if len(GPUS) > 1:
        # Spawn a process for each GPU separately and run the experiments in parallel
        processes = []
        for gpu_idx in GPUS:
            p = Process(target=run_experiments, args=(work, gpu_idx))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        # Run sequentially if only one GPU is available
        run_experiments(work, 0)
