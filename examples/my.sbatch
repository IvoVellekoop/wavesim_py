#!/bin/bash
# parameters for slurm
#SBATCH -J wavesim280           # job name, don't use spaces, keep it short
#SBATCH -c 1                    # number of cores, 1
#SBATCH --gres=gpu:ampere:2     # number of gpus 1, remove if you don't use gpu's
#SBATCH --mem=50gb              # Job memory request
#SBATCH --mail-type=END,FAIL    # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=10:00:00         # time limit 10h

# show actual node in output file, usefull for diagnostics
hostname

# Create a directory for this job on the node
cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}
# # Copy input and executable to the node
# cp -r ${SLURM_SUBMIT_DIR}/* .
# instead of copying everything, just copy the necessary files
mkdir examples
cp -r ${SLURM_SUBMIT_DIR}/examples/run_example.py examples/

mkdir logs
cp -r ${SLURM_SUBMIT_DIR}/logs/output.txt logs/

cp -r ${SLURM_SUBMIT_DIR}/wavesim .

# It's nice to have some information logged for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : "$SLURM_NNODES
echo "Number of MPI ranks         : "$SLURM_NTASKS
echo "Number of threads           : "$SLURM_CPUS_PER_TASK
echo "Number of MPI ranks per node: "$SLURM_TASKS_PER_NODE
echo "Number of threads per core  : "$SLURM_THREADS_PER_CORE
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
# Print GPU name
echo "GPU Name                    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting worker: "

# Run the job -- make sure that it terminates itself before time is up
python examples/run_example.py

# Copy output back to the master, comment with # if not used
# cp log_file.txt ${SLURM_SUBMIT_DIR}
cp -r logs ${SLURM_SUBMIT_DIR}

# Clean up on the node ! make sure you are still on the node...
#rm *
#cd ..
#rmdir ${SLURM_JOBID}

# Done.