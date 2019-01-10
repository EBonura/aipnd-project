def parse_args():
    import argparse, sys

    description = '''
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
        '''
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-dd', '--data_dir', default='flowers', metavar='', type=str, help='Set the folder where checkpoints will be saved')
    parser.add_argument('-sd', '--save_dir', default='checkpoints', metavar='', type=str, help='Set the folder where checkpoint will be saved')
    parser.add_argument('-a', '--arch', default='vgg13', metavar='', type=str, help='Set the network architecture used')
    parser.add_argument('-lr', '--learning_rate', default=.001, metavar='', type=float, help='set the learning rate')
    parser.add_argument('-h1', '--hidden_units1', default=1024, metavar='', type=int, help='set how many neurons the first hidden layer will have')
    parser.add_argument('-h2', '--hidden_units2', default=512, metavar='', type=int, help='set how many neurons the second hidden layer will have')
    parser.add_argument('-d', '--dropout', default=.2, metavar='', type=float, help='set the dropout after each hidden layer in the classifier')
    parser.add_argument('-e', '--epochs', default=10, metavar='', type=int, help='set how many epochs the network will be trained for')
    parser.add_argument('-b', '--batch', default=128, metavar='', type=int, help='set the batch size')
    parser.add_argument('-c', '--force_cpu', default=False, metavar='', type=bool, help='if set to true the model will use cpu, otherwise will use gpu when available')      
    args = parser.parse_args()
    available_arch = ('vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn','densenet121','densenet161','densenet169','densenet201','resnet101''resnet152','resnet18','resnet34','resnet50')
    
    message = f'''
        The following variables will be used:
        -------------------------------------
        data directory:       {args.data_dir}
        checkpoint directory: {args.save_dir}
        architecture:         {args.arch}
        learning rate:        {args.learning_rate}
        hidden units 1:       {args.hidden_units1}
        hidden units 2:       {args.hidden_units2}
        dropout:              {args.dropout}
        epochs:               {args.epochs}
        batch size:           {args.batch}
        force cpu:            {args.force_cpu}
        ------------------------------------
        '''
    
    if args.arch in available_arch:
        print(message)
        return args
    else:
        print(f'{args.arch} is not an valid architecture, use -h to see a full list of available architectures')
        sys.exit(0)


def load_data(args):
    # ---------------------
    # Helper function to load the data
    # ---------------------
    from torchvision import transforms, datasets
    import torch, os
    data_dir = args.data_dir
    batch_size = args.batch
    img_size = 224
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ColorJitter(.3, .3, .3),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(_mean, _std)]),
                        'valid': transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(img_size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(_mean, _std)]),
                        'test': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(img_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(_mean, _std)])}

    image_datasets = {key: datasets.ImageFolder(os.path.join(data_dir, key), data_transforms[key]) for key in data_transforms.keys()}
    dataloaders = {key: torch.utils.data.DataLoader(image_datasets[key], batch_size=batch_size, shuffle=True) for key in data_transforms.keys()}
    return dataloaders, image_datasets

def build_classifier(args, image_datasets):
    # ---------------------
    # Helper function to build the classifier
    # ---------------------
    from torchvision import models
    import torch.nn as nn
    from collections import OrderedDict
    import json
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model = models.__dict__[args.arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    input_size = model.classifier[0].in_features
    hidd1_size =  args.hidden_units1
    hidd2_size =  args.hidden_units2
    output_size = len(cat_to_name)
    dropout = args.dropout

    model.class_to_idx = image_datasets['train'].class_to_idx

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidd1_size)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p=dropout)),
                              ('fc2', nn.Linear(hidd1_size, hidd2_size)),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(p=dropout)),
                              ('output', nn.Linear(hidd2_size, output_size)),
                              ('softmax', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier    
    model.zero_grad()
    return model

def main():
    # ---------------------
    # Main training function
    # ---------------------
    
    # First thing parse the arguments
    args = parse_args()
    
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch
    from PIL import Image
    import random, os
    from datetime import datetime

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f'Using device: {device}')
    
    # Call the helper functions
    dataloaders, image_datasets = load_data(args)
    model = build_classifier(args, image_datasets)
    
    # Start training
    # ---------------------
    print('''
    Training begins
    ------------------------------------
    ''')
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    epochs = args.epochs
    logs = np.zeros(2)
    loss_log = []

    for epoch in range(epochs):
        model.train()
        running_loss, accuracy = .0, .0

        for X, y in dataloaders['train']:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            y_ = model(X)
            loss = criterion(y_, y)
            loss_log.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            
            print(f"Average Training bach loss: {running_loss/len(loss_log):.3f} -  Training batch loss: {loss.item():.3f}", end="\r")
            
            _, y_label_ = torch.max(torch.exp(y_), 1)
            accuracy += (y_label_ == y).sum().item()

        train_loss = running_loss / len(dataloaders['train'].dataset)
        train_acc = accuracy / len(dataloaders['train'].dataset)
        print(f"Epoch: {epoch+1}/{epochs}", f"Train Loss: {train_loss:.3f}", f"Train Accuracy: {train_acc:.3f}")

        model.eval()
        running_loss, accuracy = .0, .0

        with torch.no_grad():
            for X, y in dataloaders['valid']:
                X, y = X.to(device), y.to(device)
                y_ = model(X)

                loss = criterion(y_, y)
                running_loss += loss.item()

                _, y_label_ = torch.max(torch.exp(y_), 1)
                accuracy += (y_label_ == y).sum().item()

            valid_loss = running_loss / len(dataloaders['valid'].dataset)
            valid_acc = accuracy / len(dataloaders['valid'].dataset)
            print(f"Epoch: {epoch+1}/{epochs}", f"Validation Loss: {valid_loss:.3f}", f"Validation Accuracy: {valid_acc:.3f}")

            
    # Validate on the Test set
    # ---------------------
    print('''
    Validation on the test set
    ------------------------------------
    ''')
    with torch.no_grad():
        accuracy = .0
        for X, y in dataloaders['test']:
            X, y = X.to(device), y.to(device)
            y_ = model(X)

            _, y_label_ = torch.max(torch.exp(y_), 1)
            accuracy += (y_label_ == y).sum().item()

        test_acc = accuracy / len(dataloaders['test'].dataset)
        print(f"Testing Accuracy: {test_acc:.3f}")
   
    # Save the checkpoint
    # ---------------------
    savedir = args.save_dir
    if not os.path.exists(savedir):
        print(f"{savedir} doesn't exist, creating it")
        os.makedirs(savedir)
        
    curr_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint = {'classifier' : model.classifier,
                  'batch_size': args.batch,
                  'learning_rate': args.learning_rate,
                  'epochs': args.epochs,
                  'model_name': args.arch,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    file_path = f'{savedir}/{args.arch}_checkpoint_{curr_timestamp}_{test_acc:.3f}.pth'
    torch.save(checkpoint, file_path)
    print(f'Checkpoint saved in: {file_path}')
    
# Call the main function
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)