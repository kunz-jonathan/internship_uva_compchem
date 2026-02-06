from mpi4py import MPI
import subprocess
import math



inputs = []
for i in range(1, 103):
    inputs.append(
        [
            f"/df_peptide_{i}.csv",
            f"/home/kunzj/internship_uva_compchem/Data/iteration_0/binder_md_sim/jobs/job_{i}/7.analysis/mdAlignCG_1.xtc",
            f"/home/kunzj/internship_uva_compchem/Data/iteration_0/binder_md_sim/jobs/job_{i}/tpr_smaller_system.tpr"
        ]
    )
print(len(inputs))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Compute work chunk for each rank
chunk_size = math.ceil(len(inputs) / size)
start = rank * chunk_size
end = min(start + chunk_size, len(inputs))

# Process assigned inputs
for i in range(start, end):
    input_str = inputs[i]
    print(f"Rank {rank} processing: {input_str}", flush=True)
    subprocess.run(
        ["python3", "/home/kunzj/internship_uva_compchem/X.Scripts/md_sim_analysis/hydrogen_bond/worker_hbond.py", input_str[0],input_str[1],input_str[2]]
    )
