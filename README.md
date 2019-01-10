# Data Scientist Project pt. 2
Project code for Udacity's Data Scientist Nanodegree program. 

Download sample data:
```
curl -O https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
```
And extract using tar, it needs to be in the 'flowers' folder:
```
mkdir flowers
tar -xvzf flower_data.tar.gz -C flowers
```
## Training
**python train.py**

usage: train.py [-h] [-dd] [-sd] [-a] [-lr] [-h1] [-h2] [-d] [-e] [-b] [-c]

```
Train a model on a given dataset and save a checkpoint
----------------------------------------------------------------

You can use the following models:
'vgg11'          'vgg11_bn'
'vgg13'          'vgg13_bn'
'vgg16'          'vgg16_bn'
'vgg19'          'vgg19_bn'
'densenet121'    'densenet161'
'densenet169'    'densenet201'
'resnet101'      'resnet152'
'resnet18'       'resnet34'
'resnet50'

----------------------------------------------------------------


optional arguments:
  -h, --help            show this help message and exit
  -dd , --data_dir      Set the folder where checkpoints will be saved
  -sd , --save_dir      Set the folder where checkpoint will be saved
  -a , --arch           Set the network architecture used
  -lr , --learning_rate
                        set the learning rate
  -h1 , --hidden_units1
                        set how many neurons the first hidden layer will have
  -h2 , --hidden_units2
                        set how many neurons the second hidden layer will have
  -d , --dropout        set the dropout after each hidden layer in the
                        classifier
  -e , --epochs         set how many epochs the network will be trained for
  -b , --batch          set the batch size
  -c , --force_cpu      if set to true the model will use cpu, otherwise will
                        use gpu when available
```              
                        
## Prediction:             
**python predict.py -h**

usage: predict.py [-h] -p  -cd  [-k] [-n] [-c] [-r]

```
Given an image and a checkpoint, loads the model and returns flower name and class probability
----------------------------------------------------------------
Basic usage:
python predict.py -p=[image path] -cd=[checkpoint path]

Example:
python predict.py -p=flowers/valid/10/image_07101.jpg -cd=checkpoints/vgg19_checkpoint_2019-01-09-15-36-23_0.862.pth

Returns top 5 probabilities by default
              
optional arguments:
  -h, --help            show this help message and exit
  -p , --image_path     [required] Full path of the image to be predicted
  -cd , --checkpoint_path
                        [required] Full path of the checkpoint to be loaded
  -k , --top_k          Set the top K most likely classes to be returned
  -n , --category_names
                        Specify the mapping from categories to real names to
                        be used
  -c , --force_cpu      If set to true the model will use cpu, otherwise will
                        use gpu when available
  -r , --random_image   [debug] If set to true the model will load a random
                        image, --image_path will be overwritten
```
