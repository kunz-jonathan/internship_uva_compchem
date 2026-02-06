import math
import os
from pathlib import Path
from typing import List, Optional, Union

import logomaker as lm
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from Bio.PDB import PDBParser, PPBuilder
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

####================== EXPLANATION ==================####
#                                                       #
# contains plotting, data retrieval and analysis func.  #
# used over many different notebooks, thus pooled here  #
#                                                       #
####=================================================####


@beartype
def peptides_mmpbsa_energys(
    folder_name: Path, handpick_frames: Optional[set] = None
) -> dict:
    """
    handpick frames:  would give you the last 30 frames and the same frames as gmx-methods ~
    set(range(1200, 2040)).intersection(range(0, 2040, 40))
    """
    os.chdir(folder_name)
    for folder in os.listdir():
        if os.path.isfile(folder):
            continue
        os.chdir(folder)
        file = [x for x in os.listdir() if x.endswith(".csv")][0]
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Delta Energy Terms" in line:
                    start = lines.index(line)
        with open(file, "r") as f:
            lines_relevant = f.readlines()[start:]
        with open("test_results_energy_delta.csv", "w") as f:
            f.writelines(lines_relevant)
        os.chdir("..")

    peptides_energys_tables = {}

    os.chdir(folder_name)
    for folder in os.listdir():
        if os.path.isfile(folder):
            continue
        os.chdir(folder)
        df = pd.read_csv("test_results_energy_delta.csv", skiprows=1)
        peptides_energys_tables[folder.split("_")[-1]] = df
        os.chdir("..")
    peptides_energys = {}
    val = list(peptides_energys_tables.keys())
    val.sort(key=int)
    for peptides in val:
        if handpick_frames is not None:
            peptides_energys[int(peptides)] = float(
                peptides_energys_tables[peptides][
                    peptides_energys_tables[peptides]["Frame #"].isin(handpick_frames)
                ]["TOTAL"].mean()
            )
        else:
            peptides_energys[int(peptides)] = float(
                peptides_energys_tables[peptides]["TOTAL"].mean()
            )
    return peptides_energys


####====rmsd====####


@beartype
def rmsd_mean_std_calc(
    job_dir: str,
    sub_dir: str,
    file_name: str,
    file_numbers: List,
    range_frames: Optional[Union[int, slice]] = None,
) -> tuple[dict, dict]:
    """
    calculates mean & std of RMSD across trajectory,
    frames are 0 to 50 for i

    Args:
        job_dir (Path): path to jobs folder e.g. "working_dir/jobs"
        sub_dir (Path): subfolder in job_ directory containing the files e.g. "7.analysis"
        file_name (str): name of file to use,  an .xvg file is exptected e.g. "rmsd_CA_pro_to_whole_pep.xvg"
        file_numbers (List): List specifying the number of replicas e.g. ['1'] * num_jobs
        range_frames Optional(int): specifies if it should read only a range of frames, standard whole trj of 50 frames

    Returns:
        av_means, av_sds (dict) key-ligand_rank(int); mean/std(float)
    """
    av_means = {}
    av_sds = {}
    job_list = []
    for i in os.listdir(job_dir):
        if "job" in i:
            job_list.append(int(i.split("_")[1]))
    job_list.sort()

    for name, simulations in zip(job_list, file_numbers):
        all_variables = []
        sim_numbers = [int(sim_num) for sim_num in simulations.split(",")]
        filename = job_dir + "/job_" + str(name) + "/" + sub_dir + "/" + file_name

        if os.path.isfile(filename):
            for i in sim_numbers:
                data = np.loadtxt(filename, comments=["#", "@"])
                if isinstance(range_frames, slice):
                    y = data[range_frames, 1]
                else:
                    y = data[:range_frames, 1]
                all_variables.append(y)

            av_mean = np.mean(all_variables)
            av_sd = np.std(all_variables)
            av_means[name] = av_mean
            av_sds[name] = av_sd

    return av_means, av_sds


####====sasa====####


@beartype
def sasa_mean_std_calc(
    job_dir: str,
    sub_dir: str,
    file_name: str,
    file_numbers: List,
    base_path: str,
    range_frames: Optional[Union[int, slice]] = None,
) -> tuple[dict, dict]:
    """
    calculates mean & std of RMSD across trajectory

    Args:
        job_dir (Path): path to jobs folder e.g. "working_dir/jobs"
        sub_dir (Path): subfolder in job_ directory containing the files e.g. "7.analysis"
        file_name (str): name of file to use,  an .xvg file is exptected e.g. "rmsd_CA_pro_to_whole_pep.xvg"
        file_numbers (List): List specifying the number of replicas e.g. ['1'] * num_jobs
        range_frames Optional(int): specifies if it should read only a range of frames, standard whole trj of 50 frames

    Returns:
        av_means, av_sds (dict) key-ligand_rank(int); mean/std(float)
    """
    ### loading receptor allone ###

    receptor_file = (
        f"{base_path}/target_single_analysis/jobs/job_1/7.analysis/sasa_prot.xvg"
    )
    receptor_data = np.loadtxt(receptor_file, comments=["#", "@"])
    if isinstance(range_frames, slice):
        y_receptor = receptor_data[range_frames, 1]
    else:
        y_receptor = receptor_data[:range_frames, 1]

    ###======###

    av_means = {}
    av_sds = {}
    job_list = []
    for i in os.listdir(job_dir):
        if "job" in i:
            job_list.append(int(i.split("_")[1]))
    job_list.sort()

    for name, simulations in zip(job_list, file_numbers):
        all_variables = []
        sim_numbers = [int(sim_num) for sim_num in simulations.split(",")]
        filename = job_dir + "/job_" + str(name) + "/" + sub_dir + "/" + file_name

        if os.path.isfile(filename):
            for i in sim_numbers:
                data = np.loadtxt(filename, comments=["#", "@"])
                if isinstance(range_frames, slice):
                    y = data[range_frames, 1]
                else:
                    y = data[:range_frames, 1]

                all_variables.append(y)
            # substract receptor sasa

            assert len(all_variables[0]) == len(
                y_receptor
            )  # quick check if data has the same shape
            all_variables = all_variables - y_receptor

            av_mean = np.mean(all_variables)
            av_sd = np.std(all_variables)
            av_means[name] = av_mean
            av_sds[name] = av_sd

    return av_means, av_sds


####====HBOND====####


def hbond_ana_hard_cut(
    dir_donor,
    dir_acceptor,
):
    # [donor_df,acceptor_df,combined_df,count_df, [mean_both,std_both]]
    dfs = {}
    for filename in os.listdir(dir_donor):
        key = int(filename.split("_")[-1].split(".")[0])
        file_path = os.path.join(dir_donor, filename)
        dfs[key] = [pd.read_csv(file_path)]
        dfs[key][0] = dfs[key][0].drop_duplicates(
            subset=["Frame", "DonorResidue", "AcceptorResidue"]
        )

    for filename in os.listdir(dir_acceptor):
        key = int(filename.split("_")[-1].split(".")[0])
        file_path = os.path.join(dir_acceptor, filename)
        dfs[key].append(pd.read_csv(file_path))
        dfs[key][1] = dfs[key][1].drop_duplicates(
            subset=["Frame", "DonorResidue", "AcceptorResidue"]
        )

    for idx in dfs.keys():
        # combine donors and acceptors into one df
        dfs[idx].append(
            pd.concat(
                [dfs[idx][0], dfs[idx][1]], keys=["acceptor", "donor"], names=["Source"]
            ).reset_index(level="Source")
        )

        # add contacts together into one df
        dfs[idx].append(
            dfs[idx][2].groupby(["Frame", "Source"]).size().reset_index(name="count")
        )

        sum_df = dfs[idx][3].groupby("Frame", as_index=False)["count"].sum()
        sum_df["Source"] = "together"

        dfs[idx][3] = (
            pd.concat([dfs[idx][3], sum_df], ignore_index=True, sort=False)
            .sort_values(["Frame", "Source"])
            .reset_index(drop=True)
        )

        # create mean and std list
        dfs[idx].append(
            [
                np.mean(
                    dfs[idx][3][dfs[idx][3]["Source"] == "together"]["count"]
                ).item(),
                np.std(
                    dfs[idx][3][dfs[idx][3]["Source"] == "together"]["count"]
                ).item(),
            ]
        )
    return dfs


def smooth_contact(r, k=12, shift=3.25):
    """
    Smooth contact function:
    - r ≤ 3     → 1
    - 3 < r < 3.5 → smooth decay from 1 → 0
    - r ≥ 3.5   → 0
    """
    r = np.asarray(r)

    # Region fully in contact
    mask_low = r <= 3

    # Region fully out of contact
    mask_high = r >= 3.5

    # Smooth interpolation region
    mask_mid = (~mask_low) & (~mask_high)

    out = np.zeros_like(r, dtype=float)
    out[mask_low] = 1.0

    out[mask_mid] = (1 - np.tanh(k * (r[mask_mid] - shift))) / 2

    return out


def hbond_ana_switch(dir_donor, dir_acceptor, handpick_frames: Optional[set] = None):
    """
    handpick frames:  would give you the last 30 frames and the same frames as gmx-methods ~
    set(range(1200, 2040)).intersection(range(0, 2040, 40))
    """
    # [donor_df,acceptor_df,combined_df,count_df, [mean_both,std_both]]
    dfs = {}
    for filename in os.listdir(dir_donor):
        key = int(filename.split("_")[-1].split(".")[0])
        file_path = os.path.join(dir_donor, filename)
        dfs[key] = [pd.read_csv(file_path)]
        if handpick_frames is not None:
            dfs[key][0] = dfs[key][0][dfs[key][0]["Frame"].isin(handpick_frames)]
        dfs[key][0] = (
            dfs[key][0]
            .sort_values("Distance(Å)")
            .drop_duplicates(
                subset=["Frame", "DonorResidue", "AcceptorResidue"], keep="first"
            )
        )

    for filename in os.listdir(dir_acceptor):
        key = int(filename.split("_")[-1].split(".")[0])
        file_path = os.path.join(dir_acceptor, filename)
        dfs[key].append(pd.read_csv(file_path))
        if handpick_frames is not None:
            dfs[key][1] = dfs[key][1][dfs[key][1]["Frame"].isin(handpick_frames)]
        dfs[key][1] = (
            dfs[key][1]
            .sort_values("Distance(Å)")
            .drop_duplicates(
                subset=["Frame", "DonorResidue", "AcceptorResidue"], keep="first"
            )
        )

    for idx in dfs.keys():
        # combine donors and acceptors into one df
        dfs[idx].append(
            pd.concat(
                [dfs[idx][0], dfs[idx][1]], keys=["acceptor", "donor"], names=["Source"]
            ).reset_index(level="Source")
        )

        # calculate smooth distance
        dfs[idx][2]["smoothed_contacts"] = smooth_contact(
            dfs[idx][2]["Distance(Å)"].values, k=12, shift=3.25
        )

        # create new df of smoothed combined contacts per frame
        dfs[idx].append(
            dfs[idx][2].groupby("Frame")["smoothed_contacts"].sum().to_frame()
        )

        dfs[idx][3] = (
            dfs[idx][3].reindex(handpick_frames)
            if handpick_frames is not None
            else dfs[idx][3]
        )
        # dfs[idx][3] = dfs[idx][3][dfs[idx][3]["smoothed_contacts"] != 0]

        # create mean and std list
        dfs[idx].append(
            [
                np.mean(dfs[idx][3]["smoothed_contacts"]).item(),
                np.std(dfs[idx][3]["smoothed_contacts"]).item(),
            ]
        )
    return dfs


####====SBRIDGE====####


def sbridge_ana_switch(dir_donor, dir_acceptor, handpick_frames: Optional[set] = None):
    # [donor_df,acceptor_df,combined_df,count_df, [mean_both,std_both]]
    dfs = {}
    for filename in os.listdir(dir_donor):
        key = int(filename.split("_")[-1].split(".")[0])
        file_path = os.path.join(dir_donor, filename)
        dfs[key] = [pd.read_csv(file_path)]
        if handpick_frames is not None:
            dfs[key][0] = dfs[key][0][dfs[key][0]["Frame"].isin(handpick_frames)]
        dfs[key][0] = (
            dfs[key][0]
            .sort_values("Distance(Å)")
            .drop_duplicates(
                subset=["Frame", "DonorResidue", "AcceptorResidue"], keep="first"
            )
        )

    for filename in os.listdir(dir_acceptor):
        key = int(filename.split("_")[-1].split(".")[0])
        file_path = os.path.join(dir_acceptor, filename)
        dfs[key].append(pd.read_csv(file_path))
        if handpick_frames is not None:
            dfs[key][1] = dfs[key][1][dfs[key][1]["Frame"].isin(handpick_frames)]
        dfs[key][1] = (
            dfs[key][1]
            .sort_values("Distance(Å)")
            .drop_duplicates(
                subset=["Frame", "DonorResidue", "AcceptorResidue"], keep="first"
            )
        )
        dfs[key][1]

    for idx in dfs.keys():
        # combine donors and acceptors into one df
        dfs[idx].append(
            pd.concat(
                [dfs[idx][0], dfs[idx][1]], keys=["acceptor", "donor"], names=["Source"]
            ).reset_index(level="Source")
        )

        # calculate smooth distance
        dfs[idx][2]["smoothed_contacts"] = smooth_contact(
            dfs[idx][2]["Distance(Å)"].values, k=12, shift=3.75
        )

        # create new df of smoothed combined contacts per frame
        dfs[idx].append(
            dfs[idx][2].groupby("Frame")["smoothed_contacts"].sum().to_frame()
        )

        dfs[idx][3] = (
            dfs[idx][3].reindex(handpick_frames)
            if handpick_frames is not None
            else dfs[idx][3]
        )
        # dfs[idx][3] = dfs[idx][3][dfs[idx][3]["smoothed_contacts"] != 0]

        # create mean and std list
        dfs[idx].append(
            [
                np.mean(dfs[idx][3]["smoothed_contacts"]),
                np.std(dfs[idx][3]["smoothed_contacts"]),
            ]
        )
    return dfs


###====Sequence-stuff====###


@beartype
def sequence_retrieval(path: Path) -> dict:
    """
    gets all sequences from a bindcraft run

    Args:
        path to ranked folder

    Returns:
        dict: key- ligand rank(str); value - sequence (str)
    """
    pdb_folder_BAX = path
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    sequences = {}

    for file in os.listdir(pdb_folder_BAX):
        pdb_path = os.path.join(pdb_folder_BAX, file)
        structure = parser.get_structure(file, pdb_path)

        for model in structure:
            for chain in model:
                seq = "".join(
                    [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
                )
                key = str(file.split("_")[0])
                sequences[key] = seq
    return sequences


def logo_creation(sequences):
    """
    creates logo seq. representation from seq-dic
    """
    df = pd.DataFrame(columns=["peptide_id", "peptide_sequence"])
    df["peptide_id"] = sequences.keys()
    df["peptide_sequence"] = sequences.values()
    seqs = df["peptide_sequence"].apply(list)

    # Step 2: Make a DataFrame where rows = sequences, columns = positions
    alignment_df = pd.DataFrame(seqs.tolist())

    # Step 3: Count occurrences of each residue at each position
    counts_df = alignment_df.apply(lambda col: col.value_counts()).fillna(0)

    # Step 4 (optional): Convert to frequencies (Logomaker can handle counts too)
    freq_df = counts_df / counts_df.sum()
    freq_df = freq_df.transpose()
    fig = lm.Logo(
        freq_df,
        # fade_probabilities=True,
        # alpha=0.1,
        vpad=0.1,
        color_scheme="dmslogo_funcgroup",
        stack_order="small_on_top",
        font_name="sans",
        # center_values=True
    )
    return fig


@beartype
def sequence_heatmap(sequences: dict) -> None:
    """
    creates heatmap sequence representation for all sequneces

    """
    seq_values = list(sequences.values())

    # Find maximum sequence length to pad shorter sequences
    max_len = max(len(seq) for seq in seq_values)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    # Initialize frequency matrix: rows=positions, columns=amino acids
    freq_matrix = pd.DataFrame(
        0, index=range(max_len), columns=amino_acids, dtype=float
    )

    # Count residue occurrences per position
    for seq in seq_values:
        for i, aa in enumerate(seq):
            if aa in amino_acids:
                freq_matrix.at[i, aa] += 1

    # Normalize by number of sequences contributing at each position
    counts_per_pos = [
        sum(1 for seq in seq_values if len(seq) > i) for i in range(max_len)
    ]
    freq_matrix = freq_matrix.div(counts_per_pos, axis=0)

    # Plot heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(freq_matrix.T, cmap="inferno", cbar_kws={"label": "Frequency"})
    plt.xlabel("Sequence Position")
    plt.ylabel("Residue")
    plt.title(
        "Average Residue Frequency per Position Across All 100 generated Sequences"
    )
    plt.show()


###====================###


@beartype
def distribution_mean_std(
    means: dict, std: dict, xlim: tuple, ylim: tuple, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.set_context("talk")
    ax.scatter(
        means.values(),
        std.values(),
        c=means.keys(),
    )

    for i in means.keys():
        ax.annotate(
            i,
            (float(means[i]), float(std[i])),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.plot([0, 0.25], [0.05, 0.05], color='red', linestyle='--', linewidth=2)

    # ax.plot([0.25, 0.25], [0, 0.05], color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel("Mean", fontsize=16)
    ax.set_ylabel("Standard Deviation RMSD", fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.show()


def plot_rmsd_xvg_files(
    job_dir,
    sub_dir,
    file_name,
    file_numbers,
    ylim,
    xlim,
    title,
    dpi,
    sequences,
    print_mean,
) -> None:
    """
    plots rmsd for each short md of each ligand

    Args:
        job_dir (Path): path to jobs folder e.g. "working_dir/jobs"
        sub_dir (Path): subfolder in job_ directory containing the files e.g. "7.analysis"
        file_name (str): name of file to use,  an .xvg file is exptected e.g. "rmsd_CA_pro_to_whole_pep.xvg"
        file_numbers (List): List specifying the number of replicas e.g. ['1'] * num_jobs
        ylim, xlim (float): ...
        title (str)
        dpi (int)
        sequences (dict)
        print_mean (bool)
    Returns:
        None
    """
    job_list = []
    for i in os.listdir(job_dir):
        if "job" in i:
            job_list.append(int(i.split("_")[1]))
    job_list.sort()

    palette = sns.color_palette("hls", len(job_list))
    sns.set_context("talk", font_scale=1.5)
    n_files = len(job_list)

    # --- choose grid size automatically ---
    ncols = math.ceil(math.sqrt(n_files))
    nrows = math.ceil(n_files / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=dpi)
    axes = axes.flatten()

    for j, (color, (name, simulations)) in enumerate(
        zip(palette, zip(job_list, file_numbers))
    ):
        ax = axes[j]
        all_variables = []
        sim_numbers = [int(sim_num) for sim_num in simulations.split(",")]
        filename = job_dir + "/job_" + str(name) + "/" + sub_dir + "/" + file_name

        if os.path.isfile(filename):
            for i in sim_numbers:
                data = np.loadtxt(filename, comments=["#", "@"])
                x = data[:, 0]
                y = data[:, 1]
                all_variables.append(y)

            av_mean = np.mean(all_variables)
            av_sd = np.std(all_variables)
            if print_mean:
                print(f"{name}: mean = {av_mean:5.3f} nm, std = {av_sd:5.3f} nm")

            concatenate = np.stack(all_variables, axis=1)
            mean_data = concatenate.mean(axis=1)
            sd_data = concatenate.std(axis=1)

            x = np.arange(len(mean_data))
            y = mean_data
            error = sd_data / 2

            ax.plot(x, y, label=f"ligand_{name}", color=color)
            ax.fill_between(x, y - error, y + error, alpha=0.3, color=color)

            # --- subplot formatting ---
            ax.set_xlabel("Time (ns)", fontsize=18)
            ax.set_ylabel("RMSD (nm)", fontsize=18)
            ax.set_title(f"seq - {sequences[str(name)]}")

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(MultipleLocator(25))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.tick_params(which="both", width=1.5)
            ax.tick_params(which="major", direction="inout", length=6)
            ax.tick_params(which="minor", length=3, direction="in")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.legend()
        else:
            print(f"file not existing: {filename}")
            ax.set_visible(False)  # hide subplot if no file

    # Hide unused axes if grid > number of files
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


@beartype
def plot_rmsd_xvg_two_temperatures(
    job_dir_one: Path,
    job_dir_two: Path,
    sub_dir: Path,
    file_name: str,
    job_list,
    file_numbers: List,
    ylim: float,
    title: str,
    xlim: float,
    dpi: int,
    sequences: dict,
) -> None:
    """
    plots rmsd for each short md of each ligand

    Args:
        job_dir_one (Path): path to jobs folder first temp e.g. "working_dir/jobs"
        job_dir_two (Path): path to jobs folder sec. temp e.g. "working_dir/jobs"
        sub_dir (Path): subfolder in job_ directory containing the files e.g. "7.analysis"
        file_name (str): name of file to use,  an .xvg file is exptected e.g. "rmsd_CA_pro_to_whole_pep.xvg"
        file_numbers (List): List specifying the number of replicas e.g. ['1'] * num_jobs
        ylim, xlim (float): ...
        title (str)
        dpi (int)
        sequences (dict)


    Returns:
        None
    """
    job_list = []
    for i in os.listdir(job_dir_one):
        if "job" in i:
            job_list.append(int(i.split("_")[1]))
    job_list.sort()

    color = sns.color_palette("pastel")[4]
    color2 = sns.color_palette("pastel")[1]
    n_files = len(job_list)

    # --- choose grid size automatically ---
    ncols = math.ceil(math.sqrt(n_files))
    nrows = math.ceil(n_files / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=dpi)
    axes = axes.flatten()

    for j, ((name, simulations)) in enumerate(zip(job_list, file_numbers)):
        ax = axes[j]
        all_variables = []
        sim_numbers = [int(sim_num) for sim_num in simulations.split(",")]
        filename = job_dir_two + str(name) + sub_dir + "/" + file_name + ".xvg"

        if os.path.isfile(filename):
            for i in sim_numbers:
                data = np.loadtxt(filename, comments=["#", "@"])
                x = data[:, 0]
                y = data[:, 1]
                all_variables.append(y)

            av_mean = np.mean(all_variables)
            av_sd = np.std(all_variables)
            print(f"{name}: mean = {av_mean:5.3f} nm, std = {av_sd:5.3f} nm")

            concatenate = np.stack(all_variables, axis=1)
            mean_data = concatenate.mean(axis=1)
            sd_data = concatenate.std(axis=1)

            x = np.arange(len(mean_data))
            y = mean_data
            error = sd_data / 2

            ax.plot(x, y, label="350K", color=color2)
            ax.fill_between(x, y - error, y + error, alpha=0.3, color=color2)
        else:
            print(f"file not existing: {filename}")
            ax.set_visible(False)  # hide subplot if no file

    for j, (name, simulations) in enumerate(zip(job_list, file_numbers)):
        ax = axes[j]
        all_variables = []
        sim_numbers = [int(sim_num) for sim_num in simulations.split(",")]
        filename = job_dir_one + str(name) + sub_dir + "/" + file_name + ".xvg"

        if os.path.isfile(filename):
            for i in sim_numbers:
                data = np.loadtxt(filename, comments=["#", "@"])
                x = data[:, 0]
                y = data[:, 1]
                all_variables.append(y)

            av_mean = np.mean(all_variables)
            av_sd = np.std(all_variables)
            print(f"{name}: mean = {av_mean:5.3f} nm, std = {av_sd:5.3f} nm")

            concatenate = np.stack(all_variables, axis=1)
            mean_data = concatenate.mean(axis=1)
            sd_data = concatenate.std(axis=1)

            x = np.arange(len(mean_data))
            y = mean_data
            error = sd_data / 2

            ax.plot(x, y, label="310K", color=color)
            ax.fill_between(x, y - error, y + error, alpha=0.3, color=color)

            # --- subplot formatting ---
            ax.set_xlabel("Time (ns)", fontsize=12)
            ax.set_ylabel("RMSD (nm)", fontsize=12)
            ax.set_title(f"{name} - seq - {sequences[str(name)]}", fontsize=14)

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(MultipleLocator(25))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.tick_params(which="both", width=1.5)
            ax.tick_params(which="major", direction="inout", length=6)
            ax.tick_params(which="minor", length=3, direction="in")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.legend(fontsize=9)
        else:
            print(f"file not existing: {filename}")
            ax.set_visible(False)  # hide subplot if no file

    # Hide unused axes if grid > number of files
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()


@beartype
def mmmpsa_against_ranking_with_color(
    peptides_energys: dict, color_value: dict, color_label: str = "RMSD (nm)"
) -> None:
    fig, ax = plt.subplots(figsize=(35, 8))
    sns.set_context("talk")

    # Convert your hue values to a list or array
    hue_values = list(color_value.values())

    # Continuous color mapping
    scatter = sns.scatterplot(
        x=[str(x) for x in list(peptides_energys.keys())],
        y=list(peptides_energys.values()),
        hue=hue_values,
        palette="plasma",  # you can try 'coolwarm', 'mako', 'plasma', etc.
        s=300,
        ax=ax,
        legend=False,  # disable categorical legend
    )

    # Add colorbar for continuous hue
    norm = plt.Normalize(min(hue_values), max(hue_values))
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])  # only needed for Matplotlib < 3.6
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(color_label)
    ax.tick_params(
        axis="x",
        labelrotation=90,
    )
    ax.set_xticklabels(list(peptides_energys.keys()))
    print("hehehe")
    # Labels and title

    ax.set_title("")
    ax.set_xlabel("Peptides ranking")
    ax.set_ylabel("Energy (kcal/mol)")

    plt.show()


####====Umbrella====####


@beartype
def umbrella_plot(
    Name_list: List, base_path="/home/kunzj", file_path="BAX_1F16_100_peptides"
) -> None:
    File_path1 = f"{base_path}/umbrella_sampling/{file_path}/jobs/"
    data = {}
    for idx in Name_list:
        loc = File_path1 + str(idx) + "/8.analysis/"
        file_name1 = loc + "profile.xvg"
        data[idx] = np.loadtxt(file_name1, comments=["#", "@"])

    color_palette = sns.color_palette("husl", len(Name_list))

    for num, lig_idx in enumerate(data.keys()):
        x1 = data[lig_idx][:, 0]
        y1 = data[lig_idx][:, 1]

        plt.plot(
            x1 - x1[0],
            y1 - np.abs((np.max(y1[(x1 >= 2) & (x1 <= 3.5)]))),
            label=lig_idx,
            linewidth=2,
            color=color_palette[num],
        )

    plt.xticks(np.arange(0, 6, 0.5), fontsize=20)
    plt.xlim(-0.1, 1.5)  # (1.31485,3.5)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18, loc="lower right")
    plt.xlabel("ξ (nm)", fontsize=20)
    plt.ylabel("PMF (kCal/mol)", fontsize=20)
    plt.tight_layout()
    plt.show()


@beartype
def umbrella_plot_both_folder(
    Name_list: List,
    base_path="/home/kunzj",
    file_path1="BAX_1F16_100_peptides",
    file_path2="Bax_additional_peptides",
) -> None:
    data = {}
    for idx in Name_list:
        File_path1 = f"{base_path}/umbrella_sampling/{file_path1}/jobs/"
        loc = File_path1 + str(idx) + "/8.analysis/"
        file_name1 = loc + "profile.xvg"
        if os.path.isfile(file_name1):
            data[idx] = np.loadtxt(file_name1, comments=["#", "@"])
        else:
            File_path1 = f"{base_path}/umbrella_sampling/{file_path2}/jobs/"
            loc = File_path1 + str(idx) + "/8.analysis/"
            file_name1 = loc + "profile.xvg"
            data[idx] = np.loadtxt(file_name1, comments=["#", "@"])

    color_palette = sns.color_palette("husl", len(Name_list))

    for num, lig_idx in enumerate(data.keys()):
        x1 = data[lig_idx][:, 0]
        y1 = data[lig_idx][:, 1]

        plt.plot(
            x1 - x1[0],
            y1 - np.abs((np.max(y1[(x1 >= 2) & (x1 <= 3.5)]))),
            label=lig_idx,
            linewidth=2,
            color=color_palette[num],
        )

    plt.xticks(np.arange(0, 6, 0.5), fontsize=20)
    plt.xlim(-0.1, 1.5)  # (1.31485,3.5)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18, loc="lower right")
    plt.xlabel("ξ (nm)", fontsize=20)
    plt.ylabel("PMF (kCal/mol)", fontsize=20)
    plt.tight_layout()
    plt.show()


@beartype
def umbrella_samples(files: dict) -> None:
    """
    files - 'name': 'path'
    """

    def load_multi_xvg(filename):
        """Load a GROMACS .xvg file with multiple histograms in columns"""
        data = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "@", "TYPE")):
                    continue  # skip comments and metadata
                values = [float(x) for x in line.split()]
                data.append(values)
        return np.array(data)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True, sharex=True)
    axes = axes.flatten()

    for ax, (label, path) in zip(axes, files.items()):
        data = load_multi_xvg(path)
        x = data[:, 0]
        ys = data[:, 1:]

        # Plot all histograms from this file
        for i in range(ys.shape[1]):
            ax.plot(x, ys[:, i])

        ax.set_title(label)
        ax.set_xlabel("X (bin center)")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


####USELESS####
def sasa(
    job_dir,
    sub_dir,
    file_name,
    file_numbers,
    ylim,
    xlim,
    title,
    dpi,
    sequences,
    print_mean,
    atom_indices=[82, 108],  # chosen from tom's notebooks,
    select_residues=True,
) -> None:
    """
    plots rmsd for each short md of each ligand

    Args:
        job_dir (Path): path to jobs folder e.g. "working_dir/jobs"
        sub_dir (Path): subfolder in job_ directory containing the files e.g. "7.analysis"
        file_name (str): name of file to use,  an .xvg file is exptected e.g. "rmsd_CA_pro_to_whole_pep.xvg"
        file_numbers (List): List specifying the number of replicas e.g. ['1'] * num_jobs
        ylim, xlim (float): ...
        title (str)
        dpi (int)
        sequences (dict)
        print_mean (bool)
    Returns:
        None
    """
    # load job names
    job_list = []
    for i in os.listdir(job_dir):
        if "job" in i:
            job_list.append(int(i.split("_")[1]))
    job_list.sort()

    # load trajectories
    Full_peptide_trajs = {}
    for name, simulations in zip(job_list, file_numbers):
        sim_numbers = [int(sim_num) for sim_num in simulations.split(",")]
        filename = job_dir + "/job_" + str(name) + "/" + sub_dir + "/" + file_name

        Full_peptide_trajs[name] = [md.load(filename) for idx in (sim_numbers)]

    # cut first 5ns from trajectorie
    Peptide_trajs = {}

    for i, name in enumerate(Full_peptide_trajs):
        Peptide_trajs[
            job_list[i]
        ] = []  # Initialize an empty list for each key in trajs and replace name
        for j in range(len(Full_peptide_trajs[name])):
            Peptide_trajs[job_list[i]].append(
                Full_peptide_trajs[name][j][5:]
            )  # Append the sliced trajectory to the list

    fig, ax0 = plt.subplots(figsize=(8, 5))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig1, ax2 = plt.subplots(figsize=(7, 5), linewidth=2)
    fig2, ax3 = plt.subplots(figsize=(6, 4), linewidth=2)
    fig3, ax4 = plt.subplots(figsize=(6, 4), linewidth=2)

    colors = ["XKCD:black", "XKCD:bright orange", "XKCD:green", "XKCD:vivid blue"]

    mean_SASA_list = []
    error_SASA_list = []
    all_avg_data = {}

    min_length = None

    # Find the minimum length of all datasets
    for name in job_list:
        trajectories = Peptide_trajs[name]
        for trajectory in trajectories:
            if select_residues:
                top = trajectory.topology
                selected_indices = top.select(
                    "resSeq " + " ".join(map(str, atom_indices))
                )
                sliced_trajectory = trajectory.atom_slice(selected_indices)
            else:
                sliced_trajectory = trajectory.atom_slice(atom_indices)

    for i, name in enumerate(job_list):
        SASA_list = []
        single_avg_data = []
        single_sd_data = []
        trajectories = Peptide_trajs[name]

        for traj_idx, trajectory in enumerate(trajectories):
            if select_residues:
                top = trajectory.topology
                selected_indices = top.select(
                    "resSeq " + " ".join(map(str, atom_indices))
                )
                sliced_trajectory = trajectory.atom_slice(selected_indices)
            else:
                sliced_trajectory = trajectory.atom_slice(atom_indices)

            sasa = md.shrake_rupley(sliced_trajectory)
            total_sasa = sasa.sum(axis=1)
            total_sasa = total_sasa[:min_length]  # Trim to the minimum length
            SASA_list.append(total_sasa)

            avg = np.mean(total_sasa)
            sd = np.std(total_sasa)
            single_avg_data.append(avg)
            single_sd_data.append(sd)

        concatenated_SASA = np.concatenate(SASA_list, axis=0).reshape(
            len(trajectory), -1, order="F"
        )
        mean_data = concatenated_SASA.mean(axis=1)
        sd_data = concatenated_SASA.std(axis=1)
        error = sd_data / 2

        ax0.plot(sliced_trajectory.time, mean_data, label=f"{name}")
        ax0.fill_between(
            sliced_trajectory.time, mean_data - error, mean_data + error, alpha=0.3
        )
        ax1.plot(sliced_trajectory.time, mean_data, label=f"{name}")

        mean_SASA_list.append(mean_data)
        error_SASA_list.append(sd_data)

        if i == 0:
            concatenated_mean = mean_data[:, np.newaxis]
            concatenated_error = error[:, np.newaxis]
        else:
            concatenated_mean = np.concatenate(
                [concatenated_mean, mean_data[:, np.newaxis]], axis=1
            )
            concatenated_error = np.concatenate(
                [concatenated_error, error[:, np.newaxis]], axis=1
            )

        all_avg_data[name] = single_avg_data

    Name_list = ["Bax", "P1-Bax", "P2-Bax", "P3-Bax"]
    averages = [np.average(all_avg_data[name]) for name in job_list]

    # Customize ax0 and ax1
    for ax in [ax0, ax1]:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.set_xlabel("Time (ns)", fontsize=15)
        ax.set_ylabel("Total SASA (nm²)", fontsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
        fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=13)
        # ax.set_title(f'Average SASA: {Title}', fontsize=15)

    ax2.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax2.set(axisbelow=True)
    boxplot_data = [all_avg_data[name] for name in job_list]
    boxplot_labels = [name for name in job_list]
    bp = ax2.boxplot(
        boxplot_data,
        labels=boxplot_labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    ax2.scatter(
        [x + 0.05 for x in range(1, len(averages) + 1)],
        averages,
        s=160,
        marker="*",
        color="k",
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color), patch.set_alpha(0.7)

    for i, (name, y_values) in enumerate(all_avg_data.items()):
        ax2.scatter(
            [i + 0.95] * len(y_values), y_values, s=20, marker="o", color="k", zorder=3
        )

    # ax2.set_xlabel('Trajectory', fontsize=15)
    ax2.set_ylabel("Average SASA (nm²)", fontsize=20)
    # ax2.set_title(f'Average SASA Distribution: {Title}', fontsize=15)
    ax2.tick_params(axis="both", which="major", labelsize=20)
    fig1.legend = None

    for i, (name, y_values) in enumerate(all_avg_data.items()):
        ax3.scatter(
            [i + 0.97] * len(y_values), y_values, s=20, marker="o", color=colors[i]
        )
    ax3.errorbar(
        range(1, len(job_list) + 1),
        averages,
        yerr=[np.std(all_avg_data[name]) for name in job_list],
        fmt="s",
        color="k",
        capsize=10,
    )
    ax3.set_xticks(range(1, len(job_list) + 1))
    ax3.set_xticklabels([name for name in Name_list], fontsize=20)
    ax3.set_ylabel("Average SASA (nm²)", fontsize=20)
    ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=20)
    ax3.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    for i, (name, y_values) in enumerate(all_avg_data.items()):
        ax4.scatter([i + 0.97] * len(y_values), y_values, s=20, marker="o", color="k")
    # ax4.errorbar(range(1, len(job_list)+1), averages, yerr=[np.std(all_avg_data[name]) for name in job_list], fmt='s', color=[colour for colour in colors], capsize=10)
    for i, (average, name) in enumerate(zip(averages, job_list)):
        std_dev = np.std(all_avg_data[name])
        ax4.errorbar(i + 1, average, yerr=std_dev, fmt="s", color=colors[i], capsize=10)
    ax4.set_xticks(range(1, len(job_list) + 1))
    ax4.set_xticklabels([name for name in Name_list], fontsize=20)
    ax4.set_ylabel("Average SASA (nm²)", fontsize=20)
    ax4.set_yticklabels(ax4.get_yticklabels(), fontsize=20)
    ax4.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    plt.tight_layout()

    plt.show()

    return concatenated_mean, concatenated_error, averages


@beartype
def ranking_values(shape: tuple, values: List) -> pd.DataFrame:
    """
    each entry:
        [name, mean, std, ylim ]
    entries:
        - mmpbsa
        - mmpbsa scaled
        - sasa pocket
        - rmsd

    """

    # create df with columns: rank_peptide, value_mean, value_std

    df = pd.DataFrame.from_dict(values[0][1], orient="index")

    for entry in values:
        if entry[0] == "rmsd":
            df["rmsd_std"] = entry[2]
            continue
        df[f"{entry[0]}_mean"] = entry[1]
        df[f"{entry[0]}_std"] = entry[2]

    df.reset_index(inplace=True)
    df.rename(
        columns={"index": "rank_peptide", 0: f"{values[0][0]}_mean"}, inplace=True
    )

    fig, ax = plt.subplots(nrows=len(values), ncols=1, figsize=(shape[0], shape[1]))

    ax = ax.flatten()
    sns.set_context("talk")

    for i in range(len(values)):
        sns.pointplot(
            data=df, x="rank_peptide", y=f"{values[i][0]}_mean", errorbar=None, ax=ax[i]
        )
        ax[i].errorbar(
            x=range(len(df["rank_peptide"])),
            y=df[f"{values[i][0]}_mean"],
            yerr=df[f"{values[i][0]}_std"],
            fmt="none",
        )
        ax[i].set_xlim(values[i][3])

        ax[i].tick_params(axis="x", labelrotation=90)

    fig.tight_layout()

    return df


@beartype
def umbrella_plot_both_folder(
    Name_list: List,
    base_path="/home/kunzj",
    file_path1="BAX_1F16_100_peptides",
    file_path2="Bax_additional_peptides",
) -> None:
    data = {}
    for idx in Name_list:
        File_path1 = f"{base_path}/umbrella_sampling/{file_path1}/jobs/"
        loc = File_path1 + str(idx) + "/8.analysis/"
        file_name1 = loc + "profile.xvg"
        if os.path.isfile(file_name1):
            data[idx] = np.loadtxt(file_name1, comments=["#", "@"])
        else:
            File_path1 = f"{base_path}/umbrella_sampling/{file_path2}/jobs/"
            loc = File_path1 + str(idx) + "/8.analysis/"
            file_name1 = loc + "profile.xvg"
            data[idx] = np.loadtxt(file_name1, comments=["#", "@"])

    color_palette = sns.color_palette("husl", len(Name_list))

    for num, lig_idx in enumerate(data.keys()):
        x1 = data[lig_idx][:, 0]
        y1 = data[lig_idx][:, 1]

        plt.plot(
            x1 - x1[0],
            y1 - np.abs((np.max(y1[(x1 >= 2) & (x1 <= 3.5)]))),
            label=lig_idx,
            linewidth=2,
            color=color_palette[num],
        )

    plt.xticks(np.arange(0, 6, 0.5), fontsize=20)
    plt.xlim(-0.1, 1.5)  # (1.31485,3.5)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18, loc="lower right")
    plt.xlabel("ξ (nm)", fontsize=20)
    plt.ylabel("PMF (kCal/mol)", fontsize=20)
    plt.tight_layout()
    plt.show()


@beartype
def umbrella_samples(files: dict) -> None:
    """
    files - 'name': 'path'
    """

    def load_multi_xvg(filename):
        """Load a GROMACS .xvg file with multiple histograms in columns"""
        data = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "@", "TYPE")):
                    continue  # skip comments and metadata
                values = [float(x) for x in line.split()]
                data.append(values)
        return np.array(data)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True, sharex=True)
    axes = axes.flatten()

    for ax, (label, path) in zip(axes, files.items()):
        data = load_multi_xvg(path)
        x = data[:, 0]
        ys = data[:, 1:]

        # Plot all histograms from this file
        for i in range(ys.shape[1]):
            ax.plot(x, ys[:, i])

        ax.set_title(label)
        ax.set_xlabel("X (bin center)")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


