## Notice

The surrogate model is adapted from the very nicely documented [BALM repository](https://github.com/meyresearch/BALM) developed by [Gorantla, Rohan et. al. (2024) ](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02063). The model was adapted mainly to accomodate peptides as binders instead of small molecules.  <br>

The .py files containing the training scripts and the jax/torch models are here just for completeness and reference.
They are mainly copied from the forked repository of BindCraft in which they were used and deployed. 
For a more in detail understanding look at the forked repository and treat this just as a reference for what the idea is. <br>

**Reference BALM:**

@article{Gorantla2024,
  author = {Gorantla, Rohan and Gema, Aryo Pradipta and Yang, Ian Xi and Serrano-Morr{\'a}s, {\'A}lvaro and Suutari, Benjamin and Jim{\'e}nez, Jordi Ju{\'a}rez and Mey, Antonia S. J. S.},
  title = {Learning Binding Affinities via Fine-tuning of Protein and Ligand Language Models},
  year = {2024},
  doi = {10.1101/2024.11.01.621495},
  publisher = {Cold Spring Harbor Laboratory},
  journal = {bioRxiv}
}