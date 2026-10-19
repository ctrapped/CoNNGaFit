# Setting up environment on Quartz (IU)

Quartz-specific setup. For the training workflow itself see
[`../README.md`](../README.md). Info on the Quartz cluster can be found [here](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0023985).

Quartz: NVIDIA V100 / H100 GPUs.


## 1. Create conda environment 


Create a conda environment using the provided .yaml file. This primarily has the specific pytorch-cuda version needed for Quartz. Make sure you have set your conda directory set to your [SLATE project space](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0022379).

```bash
module load conda
conda env create -n connga -f environment.yaml
conda activate connga
```

Verify pytorch is correctly setup using the interactive job below.

## 2. Interactive jobs 


To run an interactive job on the debug queue run the line below, replacing `<ACCOUNT>` with your account number:
```bash
srun -A <ACCOUNT> -p h100-debug --gpus 1 --pty bash
```


To verify that the *connga* is correctly setup run:

```bash
module load conda cudatoolkit
conda activate connga
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Batch script job

Below is an example SLURM job script for running on a single GPU.


```bash
#!/bin/bash
#SBATCH --job-name=connga-train
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=h100-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1 
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -D .

# Load your modules and activate conda environment
module load conda cudatoolkit
conda activate connga

# Print module and directory info
module list
pwd

# export any environmental variables needed
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Run your program
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python my_program.py
```
