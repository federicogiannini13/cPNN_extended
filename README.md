# cPNN extended version
This repository contains the code for the experimentation shown in the paper presented at KDD Workshop on Discovering Drift Phenomena in Evolving Landscapes (DELTA 2024).

Preprint version: [link](https://aiimlab.org/pdf/events/KDD_2024_Workshop_On_Discovering_Drift_Phenomena_in_Evolving_Landscape_DELTA/Addressing%20Temporal%20Dependence%20Concept%20Drifts%20and%20Forgetting%20in%20Data%20Streams.pdf)

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
* sine_rw10: Sine RW (SRW).
* sine_rw10_mode5: Sine RW Mode (SRWM).
* weather: Weather.

<ins>Configurations:</ins>
* 1conf:
    * S1+ S2+ S1- S2- for Sine and Sine RW Mode
    * W1+ W2+ W1- W2- for Weather.
* 2conf:
    * S1+ S2- S1- S2+ for Sine and Sine RW Mode
    * W1+ W2- W1- W2+ for Weather.
* 3conf:
    * S2+ S1+ S2- S1- for Sine and Sine RW Mode
    * W2+ W1+ W2- W1- for Weather.
* 4conf:
    * S2+ S1- S2- S1+ for Sine and Sine RW Mode
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

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py**.

## Credits
https://github.com/AndreaCossu/ContinualLearning-SequentialProcessing

https://github.com/alvarolemos/pyism
