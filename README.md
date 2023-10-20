# cPNN extended version
This repository contains the code used for the experimentation shown in the paper.

## 1) Installation
execute:

`conda create -n cpnn python=3.8`

`conda activate cpnn`

`pip install -r requirements.txt`

## 2) Project structure
The project is composed of the following directories.
#### datasets
It contains the generated data streams.
Each file's name has the following structure: **\<generator\>_\<configuration\>.csv**.

<ins>Generators:</ins>
* sine_rw10_mode5: Sine RW Mode.
* weather: Weather.

<ins>Configurations:</ins>
* 1conf:
    * S1+ S2+ S1- S2- for Sine RW Mode
    * W1+ W2+ W1- W2- for Weather.
* 2conf:
    * S1+ S2- S1- S2+ for Sine RW Mode
    * W1+ W2- W1- W2+ for Weather.
* 3conf:
    * S2+ S1+ S2- S1- for Sine RW Mode
    * W2+ W1+ W2- W1- for Weather.
* 4conf:
    * S2+ S1- S2- S1+ for Sine RW Mode
    * W2+ W1- W2- W1+ for Weather.

#### models
It contains the python modules implementing cPNN, cLSTM, cGRU.
### evaluation
It contains the python modules to implement the prequential evaluation used for the experiments.
#### data
It contains the python modules implementing the data stream generator.

## 3) Evaluation
#### evaluation/test.py
It runs the prequential evaluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test`.

The execution stores the pickle files containing the results in the folder sèpecified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py**.

## Credits
https://github.com/AndreaCossu/ContinualLearning-SequentialProcessing

https://github.com/alvarolemos/pyism
