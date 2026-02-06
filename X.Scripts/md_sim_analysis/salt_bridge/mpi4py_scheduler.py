from mpi4py import MPI
import subprocess
import math

# Define your different input strings


peptide_idxs = []
for i in range(1, 103):
    peptide_idxs.append (str(i))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Compute work chunk for each rank
chunk_size = math.ceil(len(peptide_idxs) / size)
start = rank * chunk_size
end = min(start + chunk_size, len(peptide_idxs))

# Process assigned inputs
for i in range(start, end):
    input_str = peptide_idxs[i]
    print(f"Rank {rank} processing: Peptide_{input_str}", flush=True)
    subprocess.run(
        [
            "python3",
            "/home/kunzj/internship_uva_compchem/X.Scripts/md_sim_analysis/salt_bridge/worker_sbridge.py",
            input_str,
        ]
    )
