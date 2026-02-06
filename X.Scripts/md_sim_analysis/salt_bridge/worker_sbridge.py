import sys

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array

# --- Load trajectory ---
peptide_idx = sys.argv[1]

topology = f"/home/kunzj/internship_uva_compchem/Data/iteration_0/binder_md_sim/jobs/job_{peptide_idx}/top_smaller_system.gro"
trajectory = f"/home/kunzj/internship_uva_compchem/Data/iteration_0/binder_md_sim/jobs/job_{peptide_idx}/7.analysis/mdAlignCG_1.xtc"

u = mda.Universe(topology, trajectory)
u.segments.segids ='prot'
# --- update segments ---

for i in range(160,len(u.residues)):
    u.residues[i].resid += 160




peptide_segment = u.add_Segment(segid='pept')
pept = u.select_atoms(f'resid 161:{len(u.residues)}')
pept.residues.segments = peptide_segment


out_folder = (
    "/home/kunzj/internship_uva_compchem/Data/iteration_0/md_sim_analysis/sbridge/peptide_negative"
)

out_file = out_folder + f"/df_peptide_{peptide_idx}.csv"

# --- Define atom selections for salt bridges ---
# positive sidechains 
pos_sel = "(segid prot) and ((resname LYS and name NZ) or (resname ARG and (name NH1 or name NH2)) or (resname HIP and (name ND1 or name NE2)))"

# negative sidechains
neg_sel = "(segid pept) and ((resname ASP and (name OD1 or name OD2)) or (resname GLU and (name OE1 or name OE2)))"

pos_atoms = u.select_atoms(pos_sel)
neg_atoms = u.select_atoms(neg_sel)

# --- Cutoff ---
cutoff = 4.0  # Å

# --- Prepare storage ---
rows = []

# --- Loop over trajectory ---
for ts in u.trajectory:
    d = distance_array(pos_atoms.positions, neg_atoms.positions)  # NxM

    i_pos, j_neg = np.where(d <= cutoff)

    for i, j in zip(i_pos, j_neg):
        donor_resid = pos_atoms[i].residue.resid
        acceptor_resid = neg_atoms[j].residue.resid
        dist = d[i, j]
        # Angle column is excluded, fill with NaN
        rows.append([ts.frame, donor_resid, acceptor_resid, dist])

# --- Create DataFrame ---
df_saltbridges = pd.DataFrame(
    rows, columns=["Frame", "DonorResidue", "AcceptorResidue", "Distance(Å)"]
)

df_saltbridges.to_csv(out_file, index=False)
