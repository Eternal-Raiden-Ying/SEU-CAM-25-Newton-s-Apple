# ldpc
This package contains an implementation of encoders and 
decoders for the low-density parity-check (LDPC) codes
defined in the IEEE 802.11n and 802.16 families of standards.

Written by Jossy 2018

## Background info on IEEE LDPC codes

A basic knowledge of the codes in the standards is helpful:
the codes are designed via a protograph matrix.
Each entry in the matrix represents a zxz binary
cyclic shift matrix or a zxz all-zero matrix, where "-1" stands
for an all-zero matrix and any other entry represents the
magnitude of the shift (e.g. "0" is an identity matrix, i.e., 
shift by 0, while 1 is a matrix with an off-diagonal of 1s
and a 1 in the lower left corner, i.e., a matrix that performs
a left cyclic shift of its input vector). 

The 802.11n an 802.16 standards define protographs for codes
of rate 1/2, 2/3, 3/4 and 5/6. 802.11n supports z=27, 54 and 81
(where a protographs have been optimised for every combination
of rate and z), while 802.16 specifies unique protographs for
each rate that can be used with any choice of z (from z=3 upwards.)
802.16 has two optional designs (Type A and Type B) for rates
2/3 and 3/4, but only one design for rates 1/2 and 5/6.

The advantage of the IEEE 802.11n and 802.16 codes is that they
have very efficient encoders. 

## Requirements

The package is written for Python 3 and decoding functions are in C.
It's been tested on linux (CUED teaching system in DPO) and Mac OSX
High Sierra. You will need to have installed Python 3 and makes sure
you are actually using it (High Sierra comes with 2.7 by default, and
even if you've installed Python 3 you may still be using Python 2.7 
when typing "python" in a shell). You also need a compiler (gcc by
default installed on unix, if using Mac you need to install the 
"xcode command line tools"). Finally, if you want to try the automatic
testing functions, you need to install pytest (recommended approach
is to install anaconda, an package environment manager, then type
"conda install pytest"). Again, make sure pytest is using Python 3. 

## Contents and preparation

C code is in subdirectory `src/`
C executables are in subdirectory `bin/`
Python scripts classes and functions are in subdirectory `py/`
Results and data are in subdirectory `data/`

Compile the C code before first use:  
`gcc -lm -shared -fPIC -o bin/c_ldpc.so src/c_ldpc.c`  
`gcc -o bin/results2csv src/results2csv.c`

(the first of these needs to be done before decoders can be used!!)

## Description and usage

The following tools and libraries are provided:

### `py/ldpc.py`
This is the basic "code" class. It contains the following 
functions:

`ldpc.code(standard, rate, z, ptype)` initialises an ldpc object.  
`standard` is a string '802.11n' or '802.16'  
`rate` is a string (!!!) '1/2', '2/3', '3/4', '5/6'  
`z` is a number >= 3  
`ptype` is 'A' or 'B' (only needed for 802.16, rate 2/3 or 3/4)

The initialiser can be called for example as `c = ldpc.code()`  
(we will use "c" as the object name in the function descriptions below)

`c.pcmat()` returns the binary parity-check matrix 

`x = c.encode(u)` encodes the information vector u using an efficient
  encoder specialised to the IEEE standards families of LDPC codes.

`app,it = c.decode(y, dectype, corrfactor)` decodes the channel
  observation vector y to yield a-posteriori L-values app and a number
  of iterations it (at most 200, dynamically stopped using a stopping
  criterion.) dectype is either 'sumprod', 'sumprod2' or 'minsum'.
  'sumprod' is the regular sum-product algorithm
  'sumprod2' has an improved check node processing in the log domain 
  and is hence numerically more stable (and equivalent to sumprod)
  'minsum' is the min-sum algorithm. This algorithm can take an optional
   "correction factor" as an argument that can improve its performance.   
  WARNING: minsum currently NOT working, work in progress.  
  NOTE: the python function is a wrapper for underlying C functions. You
    MUST compiles these using the first `gcc` instruction above before
    importing `py/ldpc.py`

You can access code parameters using `c.K` (info length), `c.N` (codeword
  length), `c.Nv` (number of variable nodes, `= c.N`), `c.Nc` (number of
  constraint nodes), `c.Nmsg` (number of messages), `c.vdeg` (variable node
  degrees), `c.cdeg` (constraint node degrees), `c.intrlv` (code interleaver),
  `c.standard` (IEEE standard), `c.rate` (code rate string), `c.z` (z parameter
  of IEEE standard), `c.ptype` (code type for 802.16 rate 2/3 and 3/4), and
  finally `c.proto` (protograph)

### Example command-line use of `py/ldpc.py`:

NOTE: in current python version, a subdirectory must contain an empty
file `__init.py__` in order to be able to load a library from it in
command-line mode, whereas the opposite is true when not operating in 
command line mode. In the steps below, we write this file the delete it.

`$ cd ldpc`  
`$ echo " " > py/__init__.py`  
`$ python`  
`>>> import py.ldpc as ldpc`  
`>>> import numpy as np`  
`>>> c = ldpc.code()`  
`>>> c.standard`  
`'802.11n'`  
`>>> u = np.random.randint(0,2,c.K)`  
`>>> x = c.encode(u)`  
`>>> np.mod(np.matmul(x,np.transpose(c.pcmat())), 2)`  
`array([0, 0, ..., 0])`  
`>>> y = 10*(.5-x)`  
`>>> app,it = c.decode(y)`  
`>>> it`  
`0`  
`>>> np.nonzero((app<0) != x)`  
`(array([], dtype=int64),)`  
`>>> exit()`  
`$ rm py/__init__.py`  

### `py/test_ldpc.py`

Runs tests of the basic functions in ldpc.py for every combination
of parameters. The tests involve encoding a random binary word and
verifying that xH^T=0 (zero syndrome). It is not possible to design
formal test for the actual bit / word error performance of the 
decoders. These need to be examined by simulation and benchmarked
against existing published performance graphs. 
 
### `py/ldpc_awgn.py`

Runs a measurement campaign for LDPC codes on Additive White Gaussian
Noise (AWGN) channels. Usage:  
`python py/ldpc_awgn sim_id`  
where `sim_id` is a number between 1 and 36 determining the parameters
of the simulation (this was designed to work on a computer grid engine
that can run such commands in parallel with paramters 1-36). View the
file ldpc_awgn.py to see the list of parameters corresponding to sim_id
(e.g. `sim_id = 7` is 802.16, rate 1/2, z=27)

Writes all results into a file called `data/results.txt`  

A separate C programme (bin/results2csv) can be used to 
parse this file and convert it to CSV format. Finally,
a further python script `py/disp_res.py` extracts the results
from the file, sorts them back by parameters, and displays
a series of performance graphs (see below).

The directory `data/` is pre-loaded with a file `results.txt` from a 
measurement campaign conducted in September 2018. Running
such campaigns requires considerable computing power and
could take months on a single computer. 


### `disp_res.py`

Displays the performance results recorded in data/results.csv

WARNING: you must call `bin/results2csv` before calling this!
Calling `bin/results2csv` without arguments will read a file 
`data/results.txt` and wrote `data/results.csv`. An optional
command-line argument `prefix` will read and write `prefix.txt`
and `prefix.csv`, respectively, where "prefix" can include 
a directory path, e.g., `/home/user/me/ldpc/data/myresults`

Calling `python py/disp_results.py` without arguments will display
all results as python figures and wait until you've closed all
the corresponding windies.

Calling `python py/disp_results filename.pdf` will save the graphs
to the specified PDF file. 

