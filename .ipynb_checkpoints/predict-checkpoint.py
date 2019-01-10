def parse_args():
    import argparse, sys, random, os

    description = '''
        Given an image and a checkpoint, loads the model and returns flower name and class probability
        ----------------------------------------------------------------
        Basic usage: 
        python predict.py -p=[image path] -cd=[checkpoint path]
        
        Example:
        python predict.py -p=flowers/valid/10/image_07101.jpg -cd=checkpoints/vgg19_checkpoint_2019-01-09-15-36-23_0.862.pth

        Returns top 5 probabilities by default
        '''
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--image_path', metavar='', type=str, required=True, help='[required] Full path of the image to be predicted')
    parser.add_argument('-cd', '--checkpoint_path', metavar='', required=True, type=str, help='[required] Full path of the checkpoint to be loaded')
    parser.add_argument('-k', '--top_k', default=5, metavar='', type=int, help='Set the top K most likely classes to be returned')
    parser.add_argument('-n', '--category_names', default='cat_to_name.json', metavar='', type=str, help='Specify the mapping from categories to real names to be used')
    parser.add_argument('-c', '--force_cpu', default=False, metavar='', type=bool, help='If set to true the model will use cpu, otherwise will use gpu when available')
    parser.add_argument('-r', '--random_image', default=False, metavar='', type=bool, help='[debug] If set to true the model will load a random image, --image_path will be overwritten')   
    args = parser.parse_args()

    # if random_image = True a random image will be selected
    if args.random_image == True:
        rand_class = random.choice(os.listdir(f'flowers/valid/'))
        rand_pic = random.choice(os.listdir(f'flowers/valid/{rand_class}'))
        args.image_path =  f'flowers/valid/{rand_class}/{rand_pic}'    
    
    message = f'''
        The following variables will be used:
        -------------------------------------
        image path:           {args.image_path}
        checkpoint directory: {args.checkpoint_path}
        top k:                {args.top_k}
        category names json:  {args.category_names}
        force cpu:            {args.force_cpu}
        random image:         {args.random_image}
        ------------------------------------
        '''
    print(message)
    return args

def load_checkpoint(filepath):
    import os, sys, torch
    from torchvision import models
    if os.path.isfile(filepath):
        try:
            checkpoint = torch.load(filepath)
            model = models.__dict__[checkpoint['model_name']](pretrained=True)
            model.classifier = checkpoint['classifier']
            model.epochs = checkpoint['epochs']
            model.optimizer = checkpoint['optimizer']
            model.load_state_dict(checkpoint['state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']
            print('checkpoint loaded')
            return model
        except Exception as e:
           print('Cannod load the checkpoint correctly:', e)
           sys.exit(0)
    else:
        print('Provided checkpoint path does not exist')
        sys.exit(0)

def load_process_image(args):
    from torchvision import transforms
    from PIL import Image
    import os, sys
        
    if os.path.isfile(args.image_path):
        try:
            img_size = 224
            _mean = [0.485, 0.456, 0.406]
            _std = [0.229, 0.224, 0.225]
            transformations = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(_mean, _std)])
            print('Image processed')
            return transformations(Image.open(args.image_path))
        except Exception as e:
           print('Cannod load the image correctly:', e)
           sys.exit(0)
    else:
        print('Provided image path does not exist')
        sys.exit(0)

def predict(image, model, args):
    import torch    
     # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f'Using device: {device}')

    model.eval()
    model = model.to(device)
    
    X = image.unsqueeze(0)
    y_ = model(X.to(device))
    
    top_prob, top_indexes = torch.exp(y_).data.topk(args.top_k)
    
    # The classifier returns indexes, I need to convert them back to classes
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = [idx_to_class[x] for x in top_indexes.cpu().numpy().flatten()]
    
    return(top_prob.cpu().numpy().flatten(), top_classes)

def load_categories(category_names):   
    import json, os

    if os.path.isfile(category_names):
        try:
            with open(category_names, 'r') as f:
                return json.load(f)
        except Exception as e:
            print('Cannod load the json file correctly')
            sys.exit(0)
    else:
        print('Provided json path does not exist:', e)
        sys.exit(0)


def main():
    import numpy as np
    # ---------------------
    # Main predicting function
    # ---------------------
    
    # First thing parse the arguments
    args = parse_args()
    
    # Get the model first
    model = load_checkpoint(args.checkpoint_path)

    # Get the image in a proper format
    image = load_process_image(args)

    # Get the categories mapping
    cat_to_name = load_categories(args.category_names)

    # Do the prediction
    topk_prob, topk_index = predict(image, model, args)

    titles = [cat_to_name[x] for x in topk_index]
    actual_lbl = cat_to_name[args.image_path.split('/')[2]]
    predic_lbl = titles[np.argmax(topk_prob)]

    # Print everything
    print(f'\n----------------------------------')
    print(f'Top {args.top_k} probilities:  {topk_prob}')
    print(f'Top {args.top_k} indexes: {topk_index}')
    print(f'Image path: {args.image_path}')
    print(f'Original label:  {actual_lbl}, Predicted label: {predic_lbl}')
    print(f'\n----------------------------------')
    for i in range(len(titles)):
        print(f"{titles[i]} \t\t {topk_prob[i]*100:.2f}%")
    print(f'----------------------------------')
    
# Call the main function
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)