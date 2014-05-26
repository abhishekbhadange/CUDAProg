CUDAProg
========

Basic CUDA programming

The goal of the project is to compute the trail of a simulated moving object with high performance. 2D trilateration is used 
to locate the position of the object. 

GPU Algorithm: The CUDA kernel function includes the locations of three guarding points a, b and c. It also includes a very 
large set of 3-place tuples, each represented by three distances da, db and dc. The kernel program performs 2D triangulation 
for each tuple and for every 4 consecutive trilateration results, it computes the average.

Simulated Data Generation: The CPU-side of the program generates the large set of data by defining random trail to populate 
the data arrays. With this generation strategy, the program is self-verifiable - the average of every 4 consecutive values 
generated previously should correspond with the result of CUDA program.

In the end, analysis is performed by varying the size of the 3-place tuple set, and changing the block size and thread size 
parameters of the CUDA card resulting in a total combination of 28 settings.
