# OHSU Glucose

This project contains the code and data used in João Farias' Doctoral Thesis. To reproduce the results, in the order they are presented in the document, follow the steps below. Any questions with respect to the inability to reproduce the results should be directed to the author via email (`joaolcbf@gmail.com` or `joao.farias.080@ufrn.edu.br`).


# Setup

First of all, clone this repository to your local machine.

``` shell
git clone git@github.com:jotafarias13/ohsu-glucose.git      # if using ssh or
git clone https://github.com/jotafarias13/ohsu-glucose.git  # if using https
```

The project is coded in python. The recommended way to execute the scripts is using [`uv`](https://docs.astral.sh/uv/), a complete package manager for python. If using `uv`, run the following to install the project.

``` shell
uv sync
```

If you prefer not to use `uv`, you must create a virtual environment in the directory, activate it, and install dependencies like so.

``` shell
pip install -r requirements.txt
```

# Data

The data concerning virtual patient populations is located in directory `population`. The file `population_train.json` contains the patient's parameters, meals and exercises for all 10 subjects in the training population. The file `population_test.json` contains the patient's parameters, meals and exercises for all 5 subjects in the validation population.

The code used to generate the populations is also located in directory `population` and can be tested by executing the commands below. Bear in mind that population generation in stochastic, therefore each population generated will be different from one another (including the one used for the simulations).

``` shell
cd population
uv run population.py   # or
# python population.py
```


# Simulations

## Design Methodology

To replicate the results from this section, execute the commands below. Results will show up in directories `design/results_fbl`, `design/results_fbl_rbf` and `design/results_fbl_rbf_2`.

``` shell
cd design
uv run main.py   # or
# python main.py
```

## Learning Rate Optimization

To replicate the results from this section, execute the commands below. Q-tables will be saved in the directory `rl/q_tables` at each breakpoint defined in `rl/main.py`. The Q-table used for the results of the thesis is in file `rl/q_tables/q_table_final.json`.

``` shell
cd rl
uv run main.py   # or
# python main.py
```

## Training Phase Performance

To replicate the results from this subsection, execute the commands below. The argument `--population` determines which population will be simulated. Results will be saved in `results_train` for training results and `results_test` for validation results. For more details, execute `uv run main.py --help` (or the python equivalent) to check simulation options.

``` shell
uv run main.py --population train   # for the training population
uv run main.py --population test    # for the validation population
# or
# python main.py --population train 
# python main.py --population test
```

For the population graphs and the graphs with glycemia and meals/exercise presented in the thesis, execute the following. Results will be saved in directory `analysis`.

``` shell
uv run analysis.py   # or
# python analysis.py
```


## Offline RBF Pre-Training

To replicate the results from this subsection, execute the commands below. First, you need to create the reference simulations.

``` shell
cd offline
uv run main.py   # or
# python main.py
```

Results from the reference simulation will be saved in `offline/results`. Next, using the reference simulation, calculate and export the optimized weight vectors.

``` shell
cd offline          # only if not in offline dir already
uv run offline.py   # or
# python offline.py
```

Optimized weight vectors will be saved in `offline/weights`. Finally, go back to the project root and run the simulations using the pre-trained weights.

``` shell
cd ..                                 # only if still in offline dir
uv run main.py --population offline   # offline for the validation population
# or
# python main.py --population offline 
```

Results will be saved in `results_offline`. For more details, execute `uv run main.py --help` (or the python equivalent) to check simulation options.

For the population graphs and the graphs with glycemia and meals/exercise presented in the thesis, execute the following. Results will be saved in directory `analysis`.

``` shell
uv run analysis.py   # or
# python analysis.py
```

Since the last time `analysis.py` was executed there were no results for offline simulation, no graphs were generated. Now, the new graphs for offline will show up in the `analysis` directory.
