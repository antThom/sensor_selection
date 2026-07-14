# Contributing

Thank you for helping the development of `sensor_selection`! You can find the contributing instructions on this page.


## Installation

Working at APL? Make sure to follow this [guide](https://aplweb.jhuapl.edu/services/Pages/Internet-Proxy.aspx) first to ensure that you will be able to run these tools on your machine.

1. Fork the [sensor_selection](https://github.com/antThom/sensor_selection) repository on github

2. Clone your fork and make a branch of it.

Current Development is being done on `panda3_env`.

```bash
git clone https://github.com/your_username_here/sensor_seleciton
cd sensor_selection
git remote add origin https://github.com/your_username_here/sensor_seleciton
git remote add upstream https://github.com/antThom/sensor_selection
git fetch upstream panda3_env
```

3. Create a venv.

In the project's top level directory run:

```bash
python -m venv venv
./venv/Scripts/activate
```

4. Install packages with `requirements.txt`

```bash
python -m pip install -r requirements.txt
```

## While Coding

- Use [PEP8](https://peps.python.org/pep-0008/) for naming
- Add new packages? Add to requirements.txt

```bash
pip freeze >> requirements.txt
```

- Add documentation and [docstrings](https://www.geeksforgeeks.org/python/python-docstrings/)


## Before You Commit

1. Run all the tests
2. Ensure the simulation runs
```bash
python sensor_selection_simuator.py
```
3. Run `black` to format your code
4. Make sure to add a title to your commits with the `-m "Title"` flags

## External Tools

This repository uses the following tools:

- Doxygen - Documentation
- Make - Automation Utilities

To install doxygen, follow the offical [doxygen](https://www.doxygen.nl/manual/install.html) instructions. To install make, use your favorite package manager. (Windows users, try winget, scoop, or Chocolatey)


## Make Commands

Make is used in this repository for scripting automation. If other tasks need to be automated, feel free to make a new command in the Makefile.

`make clean`
Deletes build files in the repository such as `.egg` files by Panda3D and `__pycache__` files.

`make pre-commit`
Runs pre-commit tasks such as testing, formatting, and code validation
(TODO: ACTUALLY MAKE THIS)

`make doxygen-generate`
Generates doxygen documentation. Generated files can be found under `/docs/doxygen`

`make doxygen-clean`
Removes generated doxygen documentation and related files. Removes the contents of `/docs/doxygen`