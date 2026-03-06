# UvA computational chemisty group - computational peptide engineering using machine learning

As a reference what the project aims to achieve here a shortened abstract from my report at the end of the internship: <br>
<br>
<p style="text-align: justify;">
In this study we aim to design de-novo binders with the generative model, BindCraft, and
enhance its design loop with knowledge derived from atomistic Molecular Dynamics sim-
ulation. This approach will allow traversing the binder design landscape with generative 
models while guiding it towards binders with a higher binding affinity determined by MD
simulations filling the gap of explicit treatment of relevant physical metrics.
We will adapt a surrogate model to predict the binding affinity of binders by
using sequence only information mapped to physical metrics in combination with free bind-
ing energies derived by MD simulations. The model will allow us to estimate the binding
affinity of new binders without additional MD simulations and analyses.
The adaptation of our surrogate model will be validated with public available data from
the PDBbind+ database and compared against other models.
To achieve the goal of modelling physical metrics in the design process, we can integrate
the surrogate model into the binder generation of BindCraft. The substitution of explicit
calculations through surrogate models predictions is needed due to the incompatibility of
the computational cost of MD simulations and the generative design loop.
To identify the effects of the explicit modelling of physical metrics, we will repeat the whole
process two times and compare if the iterative design loops allows an improvement in our
binder designs.
</p>

<br>

## Overview

```shell
internship_uva_compchem/
    ├── X.scripts/          # containing all scripts for the **data** generation with molecular dynamics simulation using gromacs
    ├── Data/               # output folder where all the data will be generated to 
    ├── notebooks/          # containing jupyter notebooks for data analysis, figure creation etc.
    ├── surrogate_model/    # code for surrogate model (only as reference and copied from forked repo of BindCraft)
```

## Instructions

### Installation gmx_mmpbsa

1. Make sure that these following programs are installed: <br>

    * [conda/mamba](https://github.com/conda-forge/miniforge) through miniforge
    * [gromas](https://manual.gromacs.org/current/install-guide/index.html)
    * [ambertools](https://ambermd.org/GetAmber.php#ambertools)

2. downloads the .yml-file into the home directory

    ```shell
    # downloads .yml file in home_dir
    wget https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/env.yml ~/.
    ```

3. Create Conda env

    ```python
    mamba env create --file env.yml
    ```

4. Check installation 

    ```python
    # check installation ~ will install all files in the current directory
    mamba activate gmxMMPBSA
    gmx_MMPBSA_test # last step will fail if server does not allow a gui
    ```

Potential trouble shooting are mentioned in their [website](https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/installation/)

### Installation BindCraft and additional libraries

* [BindCraft](https://github.com/martinpacesa/BindCraft)

# Explanatation of directories

As a notes up: the scripts were mainly written for the in-house cluster of the computational chemistry group at UVA. Thus, ceratin changes are probably needed for different HPC

## BindCraft

### Changes to allow the use of our surrogate model

The changes one need to perform in bindcraft to register our own seq loss function

```python
# BindCraft/
#     ├── functions/
#         ├── colabdesign_utils.py    
    
from .seq_loss import add_seq_loss 

...

if advanced_settings['loss_func_seq']:
    add_seq_loss(af_model, advanced_settings["weights_seq_loss"])
```

As the esm2 module has some memory issues regarding static embedding that are getting deleted between iterations of BindCraft, one needs to clear the whole caches as otherwise the static embeddings are not newly initialized.

```python
# BindCraft/
#     ├── bindcraft.py

...

### start design loop
while True:
    
    eqx.clear_caches()
    jax.clear_caches()
```

Furthermore one needs to register the weight of our own loss function in the "settings_advanced"-file

```python
# BindCraft/
#     ├── settings_advanced/
#         ├── peptide_3stage_multimer.json

...

"loss_func_seq": true,
"weights_seq_loss": 0.8
```

The python-script that registers the surrogate model can be found here in seq_loss.py <br>
Depending on the iteration a different surrogate model needs to be loaded in the seq_loss.py

### run settings

BindCraft at its current setting can not be parralelized across multiple GPU's. However, one can run individual instances of BindCraft on different GPU's sampling for the same target. Thus, BindCraft is set up as a slurm array. <br>

## MD simulation of generated binders and target

The idea is that we create an array of slurm-jobs which splits up the generated peptides between them. For that we have a scheduler.slurm and a worker_script.sh which in the end performs the same simulation for each binder. <br>
The scheduler needs to be started from the folder ~/../binder_md_sim/scheduler.slurm , as the path are relative to this folder and slurm uses the submit folder to identify all other folders.  <br>
The target follows the same idea as the scripts of the binders and uses all files that are used for the binders. The only difference is the scheduler and the dir from which the .pdb for the simulation is taken.  

NOTE:

The path in the scheduler for the force field needs to be absolute and not relative at least from my experience and tries.

```shell
internship_uva_compchem/
    ├── X.scripts/          
        ├── binder_md_sim/
            ├── *.mdp   
            ├── data_gromacs.json   # contains parameters needed for simulation e.g. water model, box size and shape
            ├── scheduler.slurm     # splits up binders between arrays
            ├── worker_script.sh    # performs md-sim. 
        ├── target_md_sim/
            ├── scheduler.slurm     # runs the same simulation for the target but just for the target structure
            

```

## MD simulation analysis

This folder contains all scripts to analyse the md simulation of the binders.  <br>
Regarding the salt bridge and hydrogen bond the scripts need to be run twice in which one needs to change manually the folder and peptide/protein to consider once for acceptor/donor and inverse respectively. The salt bridges and hydrogen bond analyse is structured in the same way consisting of three scripts.

```shell
internship_uva_compchem/
    ├── X.scripts/          
        ├── md_sim_analysis/
            ├── bash_scripts/                   # the bash scripts perform different gmx analysis and cleaning of trajectories for further prorcesssing.
                ├── ...
            ├── hydrogen_bond/
                ├── submission_script.slurm     # submits the analysis to slurm
                ├── mpi4py_scheduler.py         # splits the binder analysis up on the cpu cores
                ├── worker_hbond.py             # performs the analysis
            ├── salt_bridge/
                ├── submission_script.slurm     # ""
                ├── mpi4py_scheduler.py         # ""
                ├── worker_sbridge.py           # ""
            ├── mmpbsa/     
                ├── file_movement.py            # moves all files neccessary for the mmpbsa analysis ~ prob. can be enhanced
                ├── calc.slurm                  # performs the mmpbsa calculation for each binder
                ├── mmpbsa.in                   # input-file for the mmpbsa calculation 
```

Certain scripts depend on each other so the prefered order of analysis would be:

1. ~/bash_scripts/align.sh ~ cleans up trajectory/needed for everything
2. ~/bash_scripts/index_creation.sh ~ needed for mmpbsa and *_smaller_system.sh
3. ~/bash_scripts/tpr_file_smaller_system.sh ~ needed for hbond-analysis
4. ~/bash_scripts/top_file_smaller_system.sh ~ needed for sbridge-analysis
5. ~/mmpbsa/file_movement.py ~ needed for mmpbsa calculation

Afterwards, all analyses can performed in any order.

## Umbrella sampling

This folder performs umbrella sampling in two steps:

1. pulling the binder from the peptide by applying an harmonic potential to the binder
2. performing short md simulations of frames while applying an harmonic potential to constrain the binder in its position

Depending on the window coverage and the smoothness of the free energy landscape one can perform additional frames by specifying them manually with umbrella_frames_additional.slurm

```shell
internship_uva_compchem/
    ├── X.scripts/          
        ├── umbrella_sampling/
            ├── helper_scripts/
                ├── file_movement.py    # create folders for each simulation of a frame 
                ├── setupUmbrella.py    # choosing configurations for frames to run the simulation, based on distance between configurations
            ├── *.mdp                                   # gromacs input files
            ├── data_gromacs.json                       # gromacs parameters 
            ├── scheduler_pull.sh                       # splits up chosen binders between array jobs
            ├── worker_pull.sh                          # runs the steered md simulation for each binder
            ├── umbrella_frames.slurm                   # runs the simulation and set up for running frames along the steered md simulation
                                                        # started from the created folder 7.umbrella by worker_pull.sh
            ├── umbrella_frames_additional.slurm        # slurm script to run additional frames if standard procedure doesnt cover the whole free energy landscape,
                                                        # requires some manual folder creation and frame specification
            ├── run_frame.sh                            # template-file for setupUmbrella.py to run frame simulation
```

## notebooks

```shell
internship_uva_compchem/
    ├── notebooks/
        ├── combining_binder_metrics.ipynb  # sourcing all data from md_sim into one file
        ├── score_creation.ipynb            # creating combined score from md_sim data
        ├── utils.py                # helper function which perform various tasks e.g. data retrieval, some simple plots 
            ...
        ├── report notebooks !!!            # TO DO: add notebooks after all figures were created for the report

```
