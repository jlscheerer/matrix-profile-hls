# Matrix Profile Datasets

The Host-Application expects the input time series data to be in a binary format and will produce the resulting matrix profile (index) in the same format. To deal with this format a [python script](../util/tsbin.py) is provided in this respository.

To **encode** custom input data (in ASCII/utf-8) run (in the `root` directory):
````bash
python3 util/tsbin.py -t double -e <Input_File_Path> -o <Output_File_Path>
````

To **decode** the result **matrix profile** run:
````bash
python3 util/tsbin.py -t double -d result.mpb -o result_mp.txt
python3 util/tsbin.py -t int -d result.mpib -o result_mpi.txt
````

For those just wanting to test the project: some exemplary datasets (in the correct format) can are provided under [data/binary](binary/)

Additionally, this repository contains further datasets used for benchmarking. Most of these datasets are from a [Repository by the Matrix Profile Foundation](https://github.com/matrix-profile-foundation/mpf-datasets). (These, however, have to be encoded as described above).