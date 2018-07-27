# Getting started

Right now the only things that are needed to run the pipeline are the script `parse_and_run.py` and the DNAProDB database
as downloaded from http://dnaprodb.usc.edu. You can download the latest database with the following command: 

```wget http://dnaprodb.usc.edu/data/dnaprodb_1.0.7z```

then unzip the database using whatever tools you prefer for doing so.

# Dependencies
`parse_and_run.py` currently relies upon keras and tensorflow to work correctly, and has only been tested with Python version 3.6.5.

# Run the model
After downloading the DNAProDB database and unzipping it, running the script is as simple as:

```
python /path/to/parse_and_run.py /path/to/collection.txt
```
Where collection.txt is the locally downloaded DNAProDB database.
This script reshapes the input data and filters DNAProDB for cocrystal structures involving multiple chains and structures with
no interacting chains with the DNA. While running, the script will periodically pipe progress messages to sys.stderr . 
