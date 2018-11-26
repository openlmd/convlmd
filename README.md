# ConvLMD, Convolutional network applied to Laser Metal Deposition.

Based on the datasets recorded from LMD process, this project provides the 
basic sctructure to develop a convolutional neural network software for process
parametrization and defects detection.

As an example, you can run this software downloading the source files and executing 
src/dilution.py, src/height.py or src/power_speed.py using as argument the path to the dataset
or extracting it inside the src folder.
If you want to use our example dataset, it is available in Zenodo platform:

https://zenodo.org/record/1556404#.W_wd3MtKjCI

<!-- Existing image-based monitoring and control approaches in laser processing
resort to embedded and PC-based platforms. Flows of data easily achieve tens
of MB per second, requiring a strong selectivity in an early stage. Few
features are gathered to control in RT usually a single parameter (e.g. laser
power) or to be used in a monitoring system for quality control. However,
gathering and processing large temporal series of data in some detail becomes
prohibitive for this kind of systems. As regards to additive manufacturing by
Laser Metal Deposition (LMD), this is of great interest since it is a long
process (e.g. lasting hours) that may accumulate important thermal and
dimensional deviations. This results in the need of reworking after the
cladding process, which means waste of material, time and energy. Moreover, it
makes difficult to ensure internal mechanical properties of produced parts. A
good example of these challenges may be found in repairing of stamping molds
for the automotive sector.

Processing data (thermal high-speed image sequences and 3D profiles) in the
cloud will bring the opportunity to gather large amounts of data and to use
machine learning techniques to extract information and to learn interrelations
between relevant process parameters. CyPLAM will rely on this approach to:

1. Adjust parameters during the process, so that deviations in a given track
can be corrected in the next layer.

2. Quality diagnosis and process reconfiguration from large series of data from
manufacturing of previous parts.

[http://openlmd.github.io/cyplam.html](http://openlmd.github.io/cyplam.html)

## Acknowledgement

This work is been supported by the European Commission through the research
project "Factories of the Future Resources, Technology, Infrastructure and
Services for Simulation and Modelling 2 (FORTISSIMO 2)", H2020 - Grant
Agreement Nº 680481.

[http://www.fortissimo-project.eu/](http://www.fortissimo-project.eu/) -->
