# Network Training

We train our network on a server running Linux system with 4 Nvidia GeForce 1080Ti GPUs, and we support the *multiple GPUs* parallel-training technique. The training script is **ONLY** tested on Linux system.

Clone the repository and enter the network training part:

    git clone https://github.com/Enigma-li/SketchCNN.git
    cd network

There are four sub-folders under the training project root ***network*** folder.
* *libs* folder contains the custom training data decoder implemented in C++ and imported as custom ops in TensorFLow framework.
* *sampleData* contains the sample training, evaluation and testing data in TFRecords format.
* *script* folder contains the network building, data loading, training and testing scripts.
* *utils* folder contains the utility functions.


## Installation

To install the training script, please first check the system and packages requirements.

### System requirement

We list the TensorFlow and Python versions, as well as some required packages:

* Our TensorFlow version is ***1.3.0***, other versions are not tested
  * We suggest users to install TensorFlow using Conda, refer to this [link](https://www.tensorflow.org/install/pip) for more details
* Our Python version is ***3.6***, other versions are note tested
* To implement the custom data decoder and monitor the training process, we require some additional packages installed:
  * OpenCV - `sudo apt-get install libopencv-dev`
  * Zlib - `sudo apt-get install zlib1g-dev`
  * opencv-python - `pip install opencv-python`

Other packages, i.e. `Numpy`, could be installed via `pip` if needed.

### Install

We first build the custom ops in *libs* folder and then configuration the Linux system to compatible with the training script. You will see the file named `custom_dataDecoder.so` generated in `libs` folder after building the ops.

* Enter *libs* folder and build the ops. Remember to change the TensorFlow source path in `build.sh` based on your system configuration.
  > cd libs <br /> ./build.sh
* Add `export PYTHONPATH="/path_to_this_repository/network"` to your `.bashrc` file if the project root is not detected.
* Add `export CUDA_DEVICE_ORDER="PCI_BUS_ID"` to your `.bashrc` file to support multi-GPU specification.

## Usage
With the project installed and custom ops built successfully, now you could try the training and testing script.

Enter the `script` folder, you will see some files whose names are beginning with "train" or "test", these will be the training and testing scripts. We accept the console parameters for training and testing configuration, just type python with `-h/--help` command to see the usage, i.e. 

    $python train_geomNet.py -h
    $python test_geomNet.py -h 

### Training

We show one example below for naive naiveNet training:

    usage: train_naiveNet.py [-h] --dbTrain DBTRAIN --dbEval DBEVAL --outDir
                         OUTDIR [--nb_gpus NB_GPUS] [--devices DEVICES]
                         [--lossId LOSSID]                             
                                                                       
    optional arguments:                                                    
      -h, --help              show this help message and exit
      --dbTrain DBTRAIN       training dataset directory
      --dbEval DBEVAL         evaluation dataset directory
      --outDir OUTDIR         otuput directory
      --nb_gpus NB_GPUS       GPU number,
      --devices DEVICES       GPU device indices
      --lossId LOSSID         training loss types - 0: naiveNet, Edn; 3: ablationNet, Edn_Eds_Ereg

💡Here, 
* GPU number, GPU device indices, lossId have default value: 1, '0', 0; modify them based on your system. 
* Output directory will be created automatically if does not exist. 
* Parameters, such as training data path should be specified by users.
* *Other training parameters are hard coded at the beginning of the script (**hyper_params**), you could change them to the values you want and it is easy to get the meaning of them.*

One typical training command for this will be:

    $python train_naiveNet.py --dbTrain=../sampleData/train --dbEval=../sampleData/eval --outDir=../output/train_naiveNet --nb_gpus=2 --devices=0,1 --lossId=0

To monitor the training process, you could use the `TensorBoard` tool as shown below:

    $cd ../output/train_naiveNet
    $tensorboard --logdir=train
    
Now you could access the training in your Browser via: `http://localhost:6006`. Be happy to use the fancy tool. :)


### Testing
We show the sample testing process of naiveNet.

The checkpoints are written into the folder `../output/train_naiveNet/savedModel`. When the training converged (about `10` epochs), you could test the network and collect the training losses values and output depth,normal maps (written to `EXR` image).

    usage: test_naiveNet.py [-h] --cktDir CKTDIR --dbTest DBTEST --outDir OUTDIR 
                        [--device DEVICE] [--lossId LOSSID] --graphName GRAPHNAME                                            
                                                                             
    optional arguments:                                                          
      -h, --help            show this help message and exit
      --cktDir CKTDIR       checkpoint directory
      --dbTest DBTEST       test dataset directory
      --outDir OUTDIR       otuput directory
      --device DEVICE       GPU device index                               
      --lossId LOSSID       training loss type - 0: naiveNet, Edn; 3: ablationNet, Edn_Eds_Ereg                                           
      --graphName           GRAPHNAME writen graph name, net.pbtxt

💡Note that:
* the `lossId` parameter should be ***SAME*** with training process
* when testing, we will write the graph definition out to one `.pbtxt` file, specify the name as you want
* Other parameters are hard coded at the beginning of the script, change them if needed.

One typical testing command will be:

    $python test_naiveNet.py --cktDir=../output/train_naiveNet/savedModel --dbTest=../sampleData/test --outDir=../output/test/test_naiveNet --device=0 --lossId=0 --graphName=SAS_naive.pbtxt

You will get the log file containing the training loss for every data item and the outputting images under folder `../output/test/test_naiveNet/out_image`.

### Freeze network to use in C++ project

We could deploy the trained networks in C++ project to develop the interactive design tool (e.g. produce some meaningful shapes by inputting 2D sketches). To use the network, we must first freeze it and prepare the configuration file (i.e. input, output nodes), we provide the tool to freeze the network in folder `utils`.

The command line parameters for this tool are:

    usage: freeze_graph_tool.py [-h] --output_dir OUTPUT_DIR --ckpt_dir CKPT_DIR
                            --ckpt_name CKPT_NAME --graph_name GRAPH_NAME
                            --net_type NET_TYPE                          
                                                                         
    optional arguments:                                                      
      -h, --help                    show this help message and exit
      --output_dir OUTPUT_DIR       output path for frozen network
      --ckpt_dir CKPT_DIR           checkpoint path
      --ckpt_name CKPT_NAME         input checkpoint name - ckpt_name.pbtxt
      --graph_name GRAPH_NAME       frozen graph name - ckpt_frozen.pb
      --net_type NET_TYPE           network type to selection output nodes 

Note that we pre-define three network types, with 0-naiveNet, 1-baselineNet, 2-GeomNet, use `--net_type` to specify the type.

One typical command will be:

    $python freeze_graph_tool.py --output_dir=../output/test/test_naiveNet --ckpt_dir=../output/train_naiveNet/savedModel -ckpt_name=SAS_naive.pbtxt --graph_name=SAS_naive_frozen.pb --net_type=0

💡 ***Special NOTE***: to use the trained network in C++ project in Windows, you should compile and build the ***SAME*** TensorFlow version, see more details in *Data generation and network deployment* part.

## Conclusion

Relying on the instructions listed above, you could train, test the networks and prepare the trained network for C++ applications in Windows.

💡Any question you could contact Changjian Li (chjili2011@gmail.com) or Hao Pan (haopan@microsoft.com) for help.

