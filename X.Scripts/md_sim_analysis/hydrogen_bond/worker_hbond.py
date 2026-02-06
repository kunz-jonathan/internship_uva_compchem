import sys

import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

out_file = sys.argv[1]
trajectory = sys.argv[2]
topology = sys.argv[3]


u = mda.Universe(topology, trajectory)

u.segments.segids ='prot'
# --- update segments ---


peptide_segment = u.add_Segment(segid='pept')
pept = u.select_atoms(f'resid 161:{len(u.residues)}')
pept.residues.segments = peptide_segment


out_folder = "/home/kunzj/internship_uva_compchem/Data/iteration_0/md_sim_analysis/hbond/peptide_acceptor"
out_file = out_folder + out_file

# names are done during gromacs and mdanalysis adds the seg_0_/seg_1_ prefix
protein_sel = "segid prot"
peptide_sel = "segid pept"

# === RUN H-BOND ANALYSIS ===
# all cutoffs were chosen based on RING specification ~ probably needs some refinement
hb = HBA(
    u,
    donors_sel=protein_sel,
    acceptors_sel=peptide_sel,
    d_h_cutoff=1.2,
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150.0,
)
hb.run()

results = hb.results.hbonds


# === PARSE RESULTS INTO RESIDUE-LEVEL TABLE ===
rows = []
for frame, donor_idx, H_idx, acceptor_idx, dist, angle in results:
    donor_atom = u.atoms[int(donor_idx)]
    acceptor_atom = u.atoms[int(acceptor_idx)]

    donor_resid = donor_atom.resid
    donor_resname = donor_atom.resname
    acceptor_resid = acceptor_atom.resid
    acceptor_resname = acceptor_atom.resname

    rows.append(
        [
            frame,
            f"{donor_resname}-{donor_resid}",
            f"{acceptor_resname}-{acceptor_resid}",
            dist,
            angle,
        ]
    )

# Create DataFrame
columns = ["Frame", "DonorResidue", "AcceptorResidue", "Distance(Å)", "Angle(°)"]
df = pd.DataFrame(rows, columns=columns)

# Save
df.to_csv(out_file, index=False)

print(f" Saved table to: {out_file}")
print(df.head())
