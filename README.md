# OHSU Glucose

This project contains the code and data used in João Farias' Doctoral Thesis. To reproduce the results, in the order they are presented in the document, follow the steps below.

# Setup

First of all, clone this repository to your local machine.

``` shell
git clone git@github.com:jotafarias13/ohsu-glucose.git      # if using ssh or
git clone https://github.com/jotafarias13/ohsu-glucose.git  # if using https
```

The project is coded in python. The recommended way to execute the script is using `uv`, a complete package manager for python. If using `uv`, run the following to install the project.

``` shell
uv sync
```

If you prefer not to use `uv`, you must create a virtual environment in the directory, activate it, and install dependencies like so.

``` shell
pip install -r requirements.txt
```


