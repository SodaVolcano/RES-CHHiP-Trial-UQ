#!/bin/bash -l
#SBATCH --job-name=chhip_uq

# SLURM RESOURCES
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1             # Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=2   # Trainer(devices=...)
#SBATCH --cpus-per-task=8     # number of cores
######SBATCH --mem-per-cpu=8GB
#SBATCH --mem=84GB
# -N is equivalent to --nodes
# -n is equivalent to --ntasks
# the default is 1G per cpu(core)
# HINT!
# --mem=0 is a special case and grants the job access to all the memory on each node!
#SBATCH --time=65:00:00
#########SBATCH --export=NONE

#################################################
module use /uwahpc/centos8/modulefiles/python
module use /uwahpc/centos8/modulefiles/devel
module load Anaconda3/2024.06 cuda/12.4
module list

#################################################
#  Note: SLURM_JOBID is a unique number for every job.
#  These are generic variables
GITREPO=https://github.com/SodaVolcano/RES-CHHiP-Trial-UQ.git
TRAIN_DATA=$MYGROUP/preprocessed_dataset.h5
SCRATCH=$MYSCRATCH/run_chhip_uq/$SLURM_JOBID
RESULTS=$MYGROUP/job_results/$SLURM_JOBID

#################################################
echo "Information about the node..."
grep -c ^processor /proc/cpuinfo
free -h --si
nvidia-smi


#!/bin/bash
echo "Variable                  Description                                      Value"
echo "-----------------------------------------------------------------------------------------"
echo "\$SLURM_JOB_ID            The Job ID                                       : $SLURM_JOB_ID"
echo "\$SLURM_SUBMIT_DIR        The path of the job submission directory         : $SLURM_SUBMIT_DIR"
echo "\$SLURM_SUBMIT_HOST       The hostname of the node used for job submission : $SLURM_SUBMIT_HOST"
echo "\$SLURM_JOB_NODELIST      Nodes assigned to the job                        : $SLURM_JOB_NODELIST"
echo "\$SLURM_CPUS_PER_TASK     Number of CPUs per task                          : $SLURM_CPUS_PER_TASK"
echo "\$SLURM_CPUS_ON_NODE      Number of CPUs on the allocated node             : $SLURM_CPUS_ON_NODE"
echo "\$SLURM_JOB_CPUS_PER_NODE Count of processors available to the job         : $SLURM_JOB_CPUS_PER_NODE"
echo "\$SLURM_CPUS_PER_GPU      Number of CPUs requested per allocated GPU       : $SLURM_CPUS_PER_GPU"
echo "\$SLURM_MEM_PER_CPU       Memory per CPU                                   : $SLURM_MEM_PER_CPU"
echo "\$SLURM_MEM_PER_GPU       Memory per GPU                                   : $SLURM_MEM_PER_GPU"
echo "\$SLURM_MEM_PER_NODE      Memory per node                                  : $SLURM_MEM_PER_NODE"
echo "\$SLURM_GPUS              Number of GPUs requested                         : $SLURM_GPUS"
echo "\$SLURM_NTASKS            Number of tasks                                  : $SLURM_NTASKS"
echo "\$SLURM_NTASKS_PER_NODE   Number of tasks per node                         : $SLURM_NTASKS_PER_NODE"
echo "\$SLURM_NTASKS_PER_SOCKET Number of tasks per socket                       : $SLURM_NTASKS_PER_SOCKET"
echo "\$SLURM_NTASKS_PER_CORE   Number of tasks per core                         : $SLURM_NTASKS_PER_CORE"
echo "\$SLURM_NTASKS_PER_GPU    Number of tasks per GPU                          : $SLURM_NTASKS_PER_GPU"
echo "\$SLURM_NNODES            Total number of nodes allocated                  : $SLURM_NNODES"
echo "\$SLURM_TASKS_PER_NODE    Number of tasks initiated per node               : $SLURM_TASKS_PER_NODE"



###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then 
    mkdir -p $SCRATCH 
fi 
echo SCRATCH is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then 
    mkdir -p $RESULTS 
fi
echo the results directory is $RESULTS

################################################
# declare the name of the output file or log file
OUTPUT=chhip_uq-$INPUT.log

#############################################
#   Copy input files to $SCRATCH
#   then change directory to $SCRATCH


# initialise git repository
cp $MYGROUP/RES-CHHiP-Trial-UQ $SCRATCH/RES-CHHiP-Trial-UQ -r
cd $SCRATCH/RES-CHHiP-Trial-UQ
ls . -al
echo "cd'ed into the directory! installing uv..."

pip install uv
uv run python3 --version
uv run python3 -c "import torch; print(torch.cuda.is_available())"
echo "finished initialising uv, copying data..."


# Copy data into $SCRATCH/RES-CHHiP-Trial-UQ
cp $TRAIN_DATA $SCRATCH/RES-CHHiP-Trial-UQ/data
echo "copied data into ${SCRATCH}"
ls $SCRATCH
echo "starting job..."

########srun --export=ALL poetry run python3 ./scripts/model_training.py
srun uv run python3 ./scripts/train_model.py unet --fold 4

#############################################
#    $OUTPUT file to the unique results dir
# note this can be a copy or move  
rm $SCRATCH/RES-CHHiP-Trial-UQ/.venv -rf
rm $SCRATCH/RES-CHHiP-Trial-UQ/data/preprocessed_dataset.h5
mv  $SCRATCH ${RESULTS}

cd $HOME

###########################
# Clean up $SCRATCH 

rm -r $SCRATCH

echo model training job finished at  `date`
