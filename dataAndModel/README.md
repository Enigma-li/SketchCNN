# Data generation and network deployment

In this part, there will be instructions for downloading data, checkpoint and sample code for deploy network in C++ applications.

## Training data

Details about data rendering, please refer to the paper and supplemental material in our [project page](http://haopan.github.io/sketchCNN.html). Totally, we generate about 30w data item, even with data compression it still has big size about ***160GB***.  We are preparing to release the data rendering codes, will update this part once ready. 

Now we provide the Google drive link for downloading training datasets:

>[Training data](https://drive.google.com/drive/folders/1eEt_5UFwx5M24QGZklWJmtCUp2md5KGh?usp=sharing)


## Trained models

We provide Google drive links for downloading the checkpoint and frozen network  files of our full network:
>[Checkpoint](https://drive.google.com/drive/folders/1X5are7bgvjAItb0JsnyyxJuzuRBU1W4f?usp=sharing) <br />
>[Frozen network](https://drive.google.com/drive/folders/15OLTP8_19dUUUXL9JN3zWkeaU3bku3kC?usp=sharing)

## network deployment

To deploy the trained network in C++ project in Windows, users must compile and build the TensorFlow libs and dlls from source using the ***SAME*** version as in network training stage. Then the source code named `trained_network.h` and `trained_network.cpp` provide one way to use the network in C++.

💡💡💡 ***Tips***:
* We provide the frozen network where you could find the sample configuration files containing the input/output nodes, channel numbers and so on, just try it.
* The first network forward pass would be time-consuming (about 2s) because of the initialization of GPU and CUDA settings. So after loading the network, please first execute the `warmup` step, all other forward passes after this `warmup` would be fast, i.e. 42ms.
* Compiling and building TensorFlow from source under Windows is time consuming (*over 2 hours*), we use **Visual Studio 2015** to build **TensorFlow 1.3**, which works for us, other configurations are **not tested**. Email us to ask for the `tensorflow.dll` if you still cannot make it.


## Conclude

In this part, we provide the training data, network checkpoint and frozen network to speedup the exploration.

💡Any question you could contact Changjian Li (chjili2011@gmail.com) or Hao Pan (haopan@microsoft.com) for help.



