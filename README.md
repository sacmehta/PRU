# Pyramidal Recurrent Units (PRU) for Language Modeling

This repository contains the source code of our paper, Pyramidal Recurrent units for language modeling.

## Downloading LM Datasets
You can download the dataset by running the following script
```
bash getdata.sh
```
This script will download the data and place in directory named **data**.

## Training PRU on the PenTree dataset
You can train PRU on the PenTree dataset (or PTB) by using following command:

```
CUDA_VISIBLE_DEVICES=0 python main.py --model PRU --g 4 --k 2 --emsize 400 --nhid 1400 --data ./data/penn 
``` 
where 
```
 --model specifies the recurrent unit (either LSTM or PRU)
 --g specifies the number of groups to be used in the grouped linear transformation, 
 --k species the number of pyramidal levels in the pyramidal transformation
 --emsize specifies the embedding layer size
 --nhid specifies the hidden layer size 
 --data specifies the location of the data.
```
Please see **main.py** for details about other supported command line arguments.

If you want to train language model using LSTM on PTB dataset, then you can do it by using the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model LSTM --emsize 400 --nhid 1000 --data ./data/penn
```

**NOTE:** Our implementation currently supports training on single GPU. However, you can easily update it to support multiple
GPUs by using DataParallel module in PyTorch.

## Testing PRU on the PenTree dataset
You can test the models using following command
```
CUDA_VISIBLE_DEVICES=0 python test.py --model PRU --g 4 --k 2 --emsize 400 --nhid 1400 --data ./data/penn --weightFile <trained model file>
```
Please see **test.py** for details about command line arguments.

## Results
With standard dropout, our model achieves following results on the PTB dataset. See our paper for more details. 

| Model | g | k | emsize | nhid | # Params | Perplexity (val) | Perplexity (test) |  
| -- | -- | -- | -- | -- | -- | -- | -- |
| LSTM | NA | NA | 400 | 1000 | 19.87 | 67.8 | 66.05 |
| LSTM | NA | NA | 400 | 1200 | 25.79 | 69.29 | 67.17 |
| LSTM | NA | NA | 400 | 1400 | 32.68 | 70.23 | 68.32 |
| -- | -- | -- | -- | -- | -- | -- | -- |
| PRU | 1 | 2 | 400 | 1000 | 18.97 | 69.99 | 68.06 |
| PRU | 2 | 2 | 400 | 1200 | 18.51 | 66.39 | 64.30 |
| PRU | 4 | 2 | 400 | 1400 | 18.90 | **64.40** | **62.62** | 

## Pre-requisite
To run this code, you need to have following libraries:
* [PyTorch](http://pytorch.org/) - We tested with v0.3.0
* Python - We tested our code with Pythonv3. If you are using Python v2, please feel free to make necessary changes to the code. 

We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.

## Acknowledgements

A large portion of this repo is borrowed from [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm) repository.

