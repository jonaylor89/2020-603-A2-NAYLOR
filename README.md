
# CMSC 603 Assignment 2 CUDA

John Naylor

### abstract

In an effort to understand the relationship between parallel algorithms and the speed up created by modifying the number of processes and cores available to these algorithm, K-nearest neighbors, which has a O(n^2) runtime complexity, was modified to be parallel by porting the implementation to the GPU with Nvidia's CUDA development kit. The modified KNN was executed with varying numbers of CUDA configurations including the thread dimensions, block size, and grid size. It was discovered that the of all the trial runs, the maximum speed up achieved was an almost 2,000 times speed up. 