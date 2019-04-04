# HIVmorbidity
Guidelines that I try to follow:


## Steps to reproduce the virtual environment and run scripts from the project

1. Clone the repo.
2. Install poetry from: https://github.com/sdispater/poetry
3. Navigate to the project directory where the ```pyproject.toml``` is.
4. Run ```poetry install```. This will install all dependencies specified in the pyproject.toml file and it will create a virtual environment.

## Version control

## Keeping the code clean and identically formatted

- Pre-commit hooks are installed and set up for the project to ensure having identically formated code across users. Immediately after installation make sure to run
```poetry run pre-commit install --install-hooks```. Then, before each commit the pre-commit hooks will be run. They will check whether the code is formated accordingly. If not, the commit will fail and the errors will be reported. The hooks which are 
run are black and flake8.

- If we want to manually run the code formater to format our code or run the linter to get usefull warnings about how our code is structured we can run ```poetry run black src/``` to format all our code
or ```poetry run flake8 src/``` to get warnings.

## Testing the code
In case we want to run all tests that we have we should run ```poetry run pytest src/```. This will run all tests and report if there are falling tests.

## Folder structure
It's suggested that we only run python files that are in the top most level in the sources directory. In other words we only treat these python files as scripts. All other python files that are
further down in the sources directory should be packed as packages and imported in the scripts as modules. An example is presented below.

```
models/
notebooks/
src/
 |
  -- script1.py
  -- script2.py
  -- package1/
        |
         -- module1.py
         -- module2.py
  -- package2/
        |
         -- module3.py
```

## Docstrings and static typing
An improved readability of the code will be achieved if we use docstrings. On the other hand, using static typing helps more than just improving readability and provides usefull warning 
messages if something seems off. One option to use is the Google docstring format and the typing library included by default in Python. An example is presented below.

```
from typing import List

def function(arg1: List[int], arg2: str):
    """Summary line of the function.
    
    Extended description of the function if needed.
    
    Args:
        arg1: Description of the first input argument.
        arg2: Description of the second input argument.

    Returns:
        What the function returns

    """
    start_of_code += 1

```
Furthermore the typing library supports all kind of types such as Dict, Tuple, Set and so on.

## Logging vs printing

The only situation where we can do ```print("Something")``` is in the top level Python files
aka scripts. Otherwise, we always use logging. The logging is included in the default
Python library. The logging package has a lot of useful features:

 - Easy to see where and when (even what line no.) a logging call is being made from.
 - You can log to files, sockets, pretty much anything, all at the same time.
 - You can differentiate your logging based on severity.
 - Print doesn't have any of these.

Also, if our project is meant to be imported by other python tools, it's bad
practice for the package to print things to stdout, since the users likely won't
know where the print messages are coming from. With logging, users of the package
can choose whether or not they want to propagate logging messages from the tool
or not.

[Stackoverlow](https://stackoverflow.com/questions/6918493/in-python-why-use-logging-instead-of-print) source
about printing vs logging.


## Notebooks

All notebooks are in the ```notebooks/``` directory and should be excluded from version control.

## Models

All saved models are in the ```models/``` directory and should be excluded from version control.

## Data

All dataset are in the ```data/``` directory and should be excluded from version control.

## Hyperparameters

All hyperparameters used to reproduce an experiment should be in the ```hyperparameters/``` directory. 


	
