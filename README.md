KeplerML
========
How to use this repository:

Recommended Prerequisite:
Filelist with lightcurve filenames.

Recommendation:
1. Collect lightcurves into a single directory.
2. Generate a filelist of the contents of that directory (the lightcurves)

    for f in \*llc.fits; do echo $f >> filelist; done
        
NOTE: For unknown reasons the code is having issues processing a whole quarter at once. Recommend splitting 
the files into at least 2 groups. An easy option is the splitting the files starting in 00 and 01
    
2. (alternate)

    for f in kplr00\*llc.fits; do echo $f >> Q??\_00filelist; done
    
    for f in kplr01\*llc.fits; do echo $f >> Q??\_01filelist; done
        
Where ?? is replaced by the quarter number.
        
Ways to use:
1. Run keplerml.py to calculate the lightcurve features, this will output a numpy array with the calculated features for each lightcurve.
    
        python keplerml.py path/to/filelist path/to/fitsfiles path/to/outputfile
2. Open Feature Calculation Example.ipynb in jupyter notebook to see examples of how to run feature calculation in a notebook.

Note: Using a 48-2.70GHz core linux computer (using 47 of the cores), processing 114,948 files took 54m:48s, which translates to 1.344 seconds to process a single file on one core. If you have less cores (most computers have 1-8 cores), multiply the number of files by the time to process a single file, and divide by the number of cores in the computer for an estimate on how long it will take to process.
