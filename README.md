[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/588524907.svg)](https://zenodo.org/badge/latestdoi/588524907)

![Alt Text](README_Front_Image.gif)
## LM-Q-NWT: A Limited Memory Quasi-Newton Approach for Multi-Objective Optimization

Implementation of the LM-Q-NWT Algorithm proposed in

[Lapucci, M., Mansueto, P. A limited memory Quasi-Newton approach for multi-objective optimization. Comput Optim Appl (2023).](
https://doi.org/10.1007/s10589-023-00454-7)

If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
@Article{Lapucci2023,
    author={Lapucci, Matteo and Mansueto, Pierluigi},
    title={A limited memory Quasi-Newton approach for multi-objective optimization},
    journal={Computational Optimization and Applications},
    year={2023},
    month={Mar},
    day={02},
    issn={1573-2894},
    doi={10.1007/s10589-023-00454-7},
    url={https://doi.org/10.1007/s10589-023-00454-7}
}
```

### Main Dependencies Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment and the Python package [nsma](https://pypi.org/project/nsma/) installed in it. For a detailed documentation of this framework, we refer the reader to its [GitHub repository](https://github.com/pierlumanzu/nsma).

For the package installation, open a terminal (Anaconda Prompt for Windows users) in the project root folder and execute the following command. Note that a Python version 3.9 or higher is required.

```
pip install nsma
```

##### Gurobi Optimizer

In order to run some parts of the code, the [Gurobi](https://www.gurobi.com/) Optimizer needs to be installed and, in addition, a valid Gurobi licence is required.

### Usage

In ```parser_management.py``` you can find all the possible arguments. Given a terminal (Anaconda Prompt for Windows users), an example of execution could be the following.

```python main.py --algs LMQNWT --probs JOS --seeds 16007 --num_trials 100 --max_time 2 --plot_pareto_front --plot_pareto_solutions --general_export --export_pareto_solutions```

### Contact

If you have any question, feel free to contact me:

[Pierluigi Mansueto](https://webgol.dinfo.unifi.it/pierluigi-mansueto/)<br>
Global Optimization Laboratory ([GOL](https://webgol.dinfo.unifi.it/))<br>
University of Florence<br>
Email: pierluigi dot mansueto at unifi dot it
