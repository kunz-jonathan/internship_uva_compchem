import os 
import shutil
from pathlib import Path
import sys


# dir of jobs e.g.: /home/kunzj/whole_pipeline/DATA/md_sim/jobs
working_dir = Path('/home/kunzj/internship_uva_compchem/Data/iteration_0/binder_md_sim/jobs')
script_dir = Path('/home/kunzj/internship_uva_compchem/X.Scripts/md_sim_analysis/mmpbsa')
dest_dir = Path('/home/kunzj/internship_uva_compchem/Data/iteration_0/md_sim_analysis/mmpbsa/jobs')

for job_path in working_dir.glob("job_*"):
    print(30*'=')
    job_idx = Path ('job_' +job_path.name.split('_')[-1])
    if int(job_path.name.split('_')[-1]) > 51:
        print(f'skipp the {job_idx}')
        continue 
    print(f'start {job_idx}')
    lig_path = job_path / '7.analysis' / "mdAlignCG_1.pdb"

    if not lig_path.exists():
        print(f'skipp the {job_idx}')
        continue    

    os.mkdir(dest_dir / job_idx)
    
    shutil.copy(working_dir / job_idx / '7.analysis' / 'mdAlignCG_1.pdb', dest_dir / job_idx / 'mdAlignCG_1.pdb')
    shutil.copy(working_dir / job_idx / '7.analysis' / 'mdAlignCG_1.xtc', dest_dir / job_idx / 'mdAlignCG_1.xtc')
    shutil.copy(working_dir / job_idx / 'index_mmpbsa.ndx', dest_dir / job_idx / 'index_mmpbsa.ndx')
    
    shutil.copy(working_dir / job_idx / '1.topology' / 'topol.top', dest_dir / job_idx / 'topol.top')
    shutil.copy(working_dir / job_idx / '1.topology' / 'topol_Protein_chain_A.itp', dest_dir / job_idx / 'topol_Protein_chain_A.itp')
    shutil.copy(working_dir / job_idx / '1.topology' / 'topol_Protein_chain_B.itp', dest_dir / job_idx / 'topol_Protein_chain_B.itp')
    shutil.copy(working_dir / job_idx / '1.topology' / 'posre_Protein_chain_B.itp', dest_dir / job_idx / 'posre_Protein_chain_B.itp')
    shutil.copy(working_dir / job_idx / '1.topology' / 'posre_Protein_chain_A.itp', dest_dir / job_idx / 'posre_Protein_chain_A.itp')
    
    shutil.copy(working_dir / job_idx / 'complex_noH.pdb', dest_dir / job_idx / 'complex_noH.pdb')
    
    shutil.copy(script_dir / 'mmpbsa.in', dest_dir / job_idx / 'mmpbsa.in')
    print(f'finished {job_idx}')
    