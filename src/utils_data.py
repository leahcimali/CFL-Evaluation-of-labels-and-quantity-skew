from src.fedclass import Client, Server
from torch.utils.data import DataLoader
from numpy import ndarray
from typing import Tuple

def shuffle_list(list_samples : int, seed : int) -> list: 
    
    """Function to shuffle the samples list

    Arguments:
        list_samples : A list of samples to shuffle
        seed : Randomization seed for reproducible results
    
    Returns:
        The shuffled list of samples 
    """

    import numpy as np

    np.random.seed(seed)

    shuffled_indices = np.arange(list_samples.shape[0])

    np.random.shuffle(shuffled_indices)

    shuffled_list = list_samples[shuffled_indices].copy()
    
    return shuffled_list


def create_label_dict(dataset : str, nn_model : str) -> tuple:
    
    """Create a dictionary of dataset samples

    Arguments:
        dataset : The name of the dataset to use (e.g 'fashion-mnist', 'mnist', or 'cifar10')
        nn_model : the training model type ('linear' or 'convolutional') 

    Returns:
        label_dict, label_test_dict : Dictionaries of data of the form {'x': [], 'y': []}
        number_of_unique_labels : Interger. The number of unique labels in the dataset

    Raises:
        Error : if the dataset name is unrecognized
    """
    
    import sys
    import numpy as np
    import torchvision
    
    import torchvision.transforms as transforms
    import os
    

    if dataset == "fashion-mnist":
        fashion_mnist = torchvision.datasets.FashionMNIST("datasets", download=True)
        (x_data, y_data) = fashion_mnist.data, fashion_mnist.targets
        fashion_mnist_test = torchvision.datasets.FashionMNIST("datasets", train = False, download=True)
        (x_test, y_test) = fashion_mnist_test.data, fashion_mnist_test.targets
        if nn_model == "convolutional":
            x_data = x_data.unsqueeze(3) # Change shape to (samples, 1, H, W)
            x_test = x_test.unsqueeze(3) # Change shape to (samples, 1, H, W)
    elif dataset == 'mnist':
        mnist = torchvision.datasets.MNIST("datasets", download=True)
        (x_data, y_data) = mnist.data, mnist.targets
        mnist_test = torchvision.datasets.MNIST("datasets", train = False, download=True)
        (x_test, y_test) = mnist_test.data, mnist_test.targets
        
        if nn_model == "convolutional":
            x_data = x_data.unsqueeze(3) # Change shape to (samples, 1, H, W)
            x_test = x_test.unsqueeze(3)
    
    elif dataset == 'kmnist':
        kmnist = torchvision.datasets.KMNIST("datasets", download=True)
        x_data, y_data = kmnist.data, kmnist.targets
        kmnist_test = torchvision.datasets.KMNIST("datasets", train = False, download=True)
        x_test, y_test = kmnist_test.data, kmnist_test.targets
        if nn_model == "convolutional":
            x_data = x_data.unsqueeze(3) # Change shape to (samples, 1, H, W)
            x_test = x_test.unsqueeze(3)
    elif dataset == "cifar10":
        if nn_model == "linear":
            raise ValueError("CIFAR-10 cannot be used with a linear model. Please use a convolutional model.")

        cifar10 = torchvision.datasets.CIFAR10("datasets", download=True)
        x_data, y_data = cifar10.data, cifar10.targets # (samples, H, W, C)
        cifar10_test = torchvision.datasets.CIFAR10("datasets", train = False, download=True)
        x_test, y_test = cifar10_test.data, cifar10_test.targets
    elif dataset in ['pathmnist','octmnist','tissuemnist']:
        import medmnist
        from medmnist import INFO, Evaluator
        '''
        Possible dataset values from medmnist:
        'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 
        'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 
        'organcmnist', 'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 
        'vesselmnist3d', 'synapsemnist3d'

        We have chosen the  following datasets for the experiment because of high number of samples 
        which is better for creating quantity skew:
        
        2D datasets of interest - Data Modality 	Tasks (# Classes/Labels) 	# Samples 	# Training / Validation / Test
        pathmnist 	Colon Pathology 	Multi-Class (9) 	107,180 	89,996 / 10,004 / 7,180
        octmnist 	Retinal OCT 	Multi-Class (4) 	109,309 	97,477 / 10,832 / 1,000
        tissuemnist 	Kidney Cortex Microscope 	Multi-Class (8) 	236,386 	165,466 / 23,640 / 47,280
        '''
        info = INFO[dataset]
        DataClass = getattr(medmnist, info['python_class'])
        if os.path.exists('datasets/{}.npz'.format(dataset)) :
            download = False
        else:
            download = True
        train = DataClass(root='datasets', split='train', download=download)
        x_data = np.array([np.array(image) for image, label in train], dtype=np.uint8)
        y_data = np.array([int(label) for image, label in train], dtype=np.uint8)
        test = DataClass(root='datasets', split='val', download=download)
        x_test = np.array([np.array(image) for image, label in test], dtype=np.uint8)
        y_test = np.array([int(label) for image, label in test], dtype=np.uint8)

    else:
        sys.exit("Unrecognized dataset. Please make sure you are using one of the following ['mnist', fashion-mnist', 'kmnist']")    

    label_dict = {}
    label_dict_test = {}
    number_of_unique_labels = len(np.unique(y_data))
    for label in range(number_of_unique_labels):
       
        label_indices = np.where(np.array(y_data) == label)[0]   
        label_samples_x = x_data[label_indices]
        label_dict[label] = label_samples_x
        label_indices_test = np.where(np.array(y_test) == label)[0]
        label_samples_x_test = x_test[label_indices_test]
        label_dict_test[label] = label_samples_x_test
        
    return label_dict, label_dict_test, number_of_unique_labels
 


def get_clients_data(num_clients : int, num_samples_by_label : int, dataset : str, nn_model : str) -> dict:
    
    """Distribute a dataset evenly accross num_clients clients. Works with datasets with 10 labels
    
    Arguments:
        num_clients : Number of clients of interest
        num_samples_by_label : Number of samples of each labels by client
        dataset: The name of the dataset to use (e.g 'fashion-mnist', 'mnist', or 'cifar10')
        nn_model : the training model type ('linear' or 'convolutional')

    Returns:
        client_dataset :  Dictionnary where each key correspond to a client index. The samples will be contained in the 'x' key and the target in 'y' key
    """
    
    import numpy as np 

    label_dict,label_dict_test, number_of_unique_labels = create_label_dict(dataset, nn_model)
    

    clients_dictionary = {}
    client_dataset = {}

    for client in range(num_clients):
        
        clients_dictionary[client] = {'train': {}, 'test': {}}    
        
        for label in range(number_of_unique_labels):
            if num_samples_by_label > (len(label_dict[label])//num_clients):
                
                num_samples_by_label = len(label_dict[label])//num_clients

            clients_dictionary[client]['train'][label]= label_dict[label][client*num_samples_by_label:(client+1)*num_samples_by_label]
            test_num_samples_by_label = len(label_dict_test[label]) // num_clients

            clients_dictionary[client]['test'][label]= label_dict_test[label][client*test_num_samples_by_label:(client+1)*test_num_samples_by_label]
    
    for client in range(num_clients):
    
        client_dataset[client] = {}    
    
        client_dataset[client]['x'] = np.concatenate([clients_dictionary[client]['train'][label] for label in range(number_of_unique_labels)], axis=0)
    
        client_dataset[client]['y'] = np.concatenate([[label]*len(clients_dictionary[client]['train'][label]) for label in range(number_of_unique_labels)], axis=0)
        
        client_dataset[client]['x_test'] = np.concatenate([clients_dictionary[client]['test'][label] for label in range(number_of_unique_labels)], axis=0)
        
        client_dataset[client]['y_test'] = np.concatenate([[label]*len(clients_dictionary[client]['test'][label]) for label in range(number_of_unique_labels)], axis=0)
    
    return client_dataset



def rotate_images(client: Client, rotation: int) -> None:
    
    """ Rotate a Client's images, used for ``concept shift on features''
    
    Arguments:
        client : A Client object whose dataset images we want to rotate
        rotation : the rotation angle to apply  0 < angle < 360
    """
    
    import numpy as np

    images = client.data['x']
    test_images = client.data['x_test']
    if rotation > 0 :

        rotated_images = []
        rotated_images_test = []
        for img in images:
    
            orig_shape = img.shape             
            rotated_img = np.rot90(img, k=rotation//90)  # Rotate image by specified angle 
            rotated_img = rotated_img.reshape(*orig_shape)
            rotated_images.append(rotated_img)   
        
        client.data['x'] = np.array(rotated_images)
        
        for img_test in test_images:    
            
            orig_shape_test = img.shape             
            rotated_img_test = np.rot90(img_test, k=rotation//90)  # Rotate image by specified angle 
            rotated_img_test = rotated_img_test.reshape(*orig_shape_test)
            rotated_images_test.append(rotated_img_test)
            
        client.data['x_test'] = np.array(rotated_images_test)
    return

import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    """Custom dataset to apply transformations."""
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = self.x_data[idx]
        label = self.y_data[idx]
        #if sample.shape[0] == 3:  # This implies CIFAR-10's RGB data
        #    sample = transforms.ToPILImage()(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
def data_transformation(row_exp : dict)-> tuple:
    '''
    Create transformation relative to the dataset and nn_model type
    Arguments:
        row_exp : The current experiment's global parameters
    Returns:
        tuple: A tuple containing three elements:
            - train_transform (transforms.Compose): Transformations applied 
              to the training dataset.
            - val_transform (transforms.Compose): Transformations applied 
              to the validation dataset.
            - test_transform (transforms.Compose): Transformations applied 
              to the test dataset.
    '''
    if row_exp['dataset'] == 'cifar10' : 
        CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            #transforms.ColorJitter(brightness=0.5, hue=0.5),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])      
    elif row_exp['nn_model'] == 'convolutional': #dataset of mnist type
        # For other datasets
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ]) 
    else : #linear + mnist type dataset
         # For other datasets
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
        transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
        transforms.ToTensor()
        ])     
    return train_transform, val_transform, test_transform 
    
def data_preparation(client: Client, row_exp: dict) -> None:
    """Saves Dataloaders of train and test data in the Client attributes 
    
    Arguments:
        client : The client object to modify
        row_exp : The current experiment's global parameters
    """
    
    import torch 
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    import torchvision.transforms as transforms
    import numpy as np  # Import NumPy for transpose operation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split into train, validation, and test sets
    x_train, y_train = client.data['x'], client.data['y']
    #Spit test set in validation and test
    
    x_val, x_test, y_val, y_test = train_test_split(
        client.data['x_test'], client.data['y_test'], test_size=0.7, 
        random_state=row_exp['seed'], stratify=client.data['y_test'])
    
    # Define data augmentation transforms
    train_transform, val_transform, test_transform = data_transformation(row_exp)

    # Create datasets with transformations
    train_dataset = CustomDataset(x_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(x_val, y_val, transform=val_transform)
    test_dataset = CustomDataset(x_test, y_test, transform=test_transform)
    batch_size = 256 if row_exp['dataset'] == 'cifar10' else 128
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Store DataLoaders in the client object
    setattr(client, 'data_loader', {'train': train_loader, 'val': validation_loader, 'test': test_loader})
    setattr(client, 'train_test', {'x_train': x_train, 'x_val': x_val, 'x_test': x_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test})

    return



def get_dataset_heterogeneities(row_exp) -> dict:

    """
    Retrieves the "skew" and "ratio" attributes of a given heterogeneity type

    Arguments:
        heterogeneity_type : The label of the heterogeneity scenario (labels-distribution-skew, concept-shift-on-labels, quantity-skew)
    Returns:
        dict_params: A dictionary of the form {<het>: []} where <het> is the applicable heterogeneity type 
    """
    dict_params = {}
    heterogeneity_type = row_exp['heterogeneity_type']
    if heterogeneity_type == 'labels-distribution-skew':
        dict_params['ratios'] = [
    [0.05, 0.2875, 0.525, 0.7625, 0.95, 0.7625, 0.525, 0.2875, 0.10,0.05],  # Normal distribution
    [0.95, 0.7125, 0.475, 0.2375, 0.05, 0.1, 0.2375, 0.475, 0.7125, 0.95],  # Complementary to normal distribution
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],       # Left-skewed distribution
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]        # Right-skewed distribution
    ]
    
    elif heterogeneity_type == 'concept-shift-on-labels':
        if row_exp['dataset'] == 'octmnist':
            dict_params['swaps'] = [(0,1),(0,2),(1,3),(2,3)]
        else : 
            dict_params['swaps'] = [(1,7),(2,7),(4,7),(3,8)]

    elif  heterogeneity_type == 'quantity-skew' :
        dict_params['skews'] = [0.1,0.2,0.6,1]
    


    return dict_params
    

def setup_experiment(row_exp: dict) -> Tuple[Server, list]:

    """ Setup function to create and personalize client's data 

    Arguments:
        row_exp : The current experiment's global parameters


    Returns: 
        fl_server, list_clients: a nn model used the server in the FL protocol, a list of Client Objects used as nodes in the FL protocol

    """

    from src.models import GenericConvModel,GenericLinearModel, SimpleConvModel
    from src.utils_fed import init_server_cluster
    import torch
    
    list_clients = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(row_exp['seed'])
    # image size, number of channels, number of classes
    imgs_params = {'mnist': (28,1,10) , 'fashion-mnist': (28,1,10), 'kmnist': (28,1,10), 'pathmnist':(28,3,9)
                   , 'tissuemnist':(28,1,8),'octmnist':(28,1,4), 'cifar10': (32,3,10)}

    if row_exp['nn_model'] == "linear":
        
        fl_server = Server(GenericLinearModel(in_size=imgs_params[row_exp['dataset']][0])) 
    
    elif row_exp['nn_model'] == "convolutional": 
        if row_exp['dataset'] in ['tissuemnist','octmnist']:
            fl_server = Server(SimpleConvModel(in_size=imgs_params[row_exp['dataset']][0], n_channels=imgs_params[row_exp['dataset']][1],num_classes=imgs_params[row_exp['dataset']][2]))
        else : 
            fl_server = Server(GenericConvModel(in_size=imgs_params[row_exp['dataset']][0], n_channels=imgs_params[row_exp['dataset']][1],num_classes=imgs_params[row_exp['dataset']][2]))
       

    fl_server.model.to(device)

    dict_clients = get_clients_data(row_exp['num_clients'],
                                    row_exp['num_samples_by_label'],
                                    row_exp['dataset'],
                                    row_exp['nn_model'])    
    
    for i in range(row_exp['num_clients']):

        list_clients.append(Client(i, dict_clients[i]))

    list_clients = add_clients_heterogeneity(list_clients, row_exp)
    
    if row_exp['exp_type'] == "ifca":

        init_server_cluster(fl_server, list_clients, row_exp, imgs_params[row_exp['dataset']])

    return fl_server, list_clients



def add_clients_heterogeneity(list_clients: list, row_exp: dict) -> list:
    """ Utility function to apply the relevant heterogeneity classes to each client
    
    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    Returns:
        The updated list of clients
    """

    dict_params = get_dataset_heterogeneities(row_exp)
    # Concept shift on features
    if row_exp['heterogeneity_type']  == "concept-shift-on-features": # rotations
        if row_exp['skew'] == "quantity-skew-type-1":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 1) 
        elif row_exp['skew'] == "quantity-skew-type-2":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 2) 
        # if skew as value none or other value will not apply skewness
        if row_exp['dataset'] == "pathmnist":
            list_clients = apply_channel_invertion(list_clients, row_exp)
        elif row_exp['dataset'] in ['tissuemnist','octmnist']:
            list_clients = apply_invert_greyscale_and_zoom(list_clients, row_exp)
        else :
            list_clients = apply_rotation(list_clients, row_exp)
        if row_exp['skew'] == "label-skew":
            dict_params = get_dataset_heterogeneities(row_exp)    
            list_clients = apply_labels_skew(list_clients, row_exp, # less images of certain labels
                                          dict_params['ratios'])
    # Concept shift on labels    
    elif row_exp['heterogeneity_type'] == "concept-shift-on-labels": #label swaps
        if row_exp['skew'] == "quantity-skew-type-1":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 1) 
        elif row_exp['skew'] == "quantity-skew-type-2":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 2) 
        # if skew as value none or other value will not apply skewness
        list_clients = apply_label_swap(list_clients, row_exp, dict_params['swaps'])
        if row_exp['skew'] == "label-skew":
            dict_params = get_dataset_heterogeneities(row_exp)    
            list_clients = apply_labels_skew(list_clients, row_exp, # less images of certain labels
                                          dict_params['ratios'])
    # Features distribution skew        
    elif row_exp['heterogeneity_type'] == "features-distribution-skew": #change image qualities
        if row_exp['skew'] == "quantity-skew-type-1":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 1) 
        elif row_exp['skew'] == "quantity-skew-type-2":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 2) 
        # if skew as value none or other value will not apply skewness
        list_clients = apply_features_skew(list_clients, row_exp)
        if row_exp['skew'] == "label-skew":
            dict_params = get_dataset_heterogeneities(row_exp)    
            list_clients = apply_labels_skew(list_clients, row_exp, # less images of certain labels
                                          dict_params['ratios'])
    # Labels distribution skew
    elif row_exp['heterogeneity_type'] == "labels-distribution-skew":
        list_clients = apply_labels_skew(list_clients, row_exp, dict_params['skews'], # less images of certain labels
                                          dict_params['ratios'])
    # Quantity skew
    elif row_exp['heterogeneity_type'] == "quantity-skew": #less images altogether for certain clients
        list_clients = apply_quantity_skew(list_clients, row_exp, dict_params['skews']) 
    elif row_exp['heterogeneity_type'] == "None":
        if row_exp['skew'] == "quantity-skew-type-1":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 1) 
        elif row_exp['skew'] == "quantity-skew-type-2":
            list_clients = apply_quantity_skew(list_clients, row_exp, [0.05,0.2,1,2],skew_type = 2) 
        elif row_exp['skew'] == "label-skew":
            dict_params = get_dataset_heterogeneities(row_exp)    
            list_clients = apply_labels_skew(list_clients, row_exp, # less images of certain labels
                                          dict_params['ratios'])
        for client in list_clients:
            data_preparation(client, row_exp)

    return list_clients



def apply_label_swap(list_clients : list, row_exp : dict, list_swaps : list) -> list:
    
    """ Utility function to apply label swaps on Client images

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
        list_swap : List containing the labels to swap by heterogeneity class
    Returns :
        Updated list of clients
    
    """
    n_swaps_types = len(list_swaps)
    
    n_clients_by_swaps_type = row_exp['num_clients'] // n_swaps_types
    
    for i in range(n_swaps_types):
        
        start_index = i * n_clients_by_swaps_type
        end_index = (i + 1) * n_clients_by_swaps_type

        list_clients_swapped = list_clients[start_index:end_index]

        for client in list_clients_swapped:
            
            client = swap_labels(list_swaps[i],client, str(list_swaps[i]))
            
            data_preparation(client, row_exp)

        list_clients[start_index:end_index] = list_clients_swapped

    list_clients  = list_clients[:end_index]

    return list_clients




def apply_rotation(list_clients : list, row_exp : dict) -> list:

    """ Utility function to apply rotation 0,90,180 and 270 to 1/4 of Clients 

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients
    """
    
    n_rotation_types = 4
    n_clients_by_rotation_type = row_exp['num_clients'] // n_rotation_types #TODO check edge cases where n_clients < n_rotation_types

    for i in range(n_rotation_types):
        
        start_index = i * n_clients_by_rotation_type
        end_index = (i + 1) * n_clients_by_rotation_type

        list_clients_rotated = list_clients[start_index:end_index]

        for client in list_clients_rotated:

            rotation_angle = (360 // n_rotation_types) * i

            rotate_images(client , rotation_angle)
            
            data_preparation(client, row_exp)
            setattr(client,'heterogeneity_class', f"rot_{rotation_angle}")

        list_clients[start_index:end_index] = list_clients_rotated

    list_clients  = list_clients[:end_index]

    return list_clients

def apply_invert_greyscale_and_zoom(list_clients : list, row_exp : dict) -> list:

    """ Utility function to apply normal, greyscale inversion, normal with zoom and greyscale inversion with zoom to 1/4 of Clients 

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients
    """
    import numpy as np
    
    number_of_client_per_transformation = row_exp['num_clients'] // 4

    for i in range(0,4):
        
        start_index = i * number_of_client_per_transformation
        end_index = (i + 1) * number_of_client_per_transformation

        list_clients_transformed = list_clients[start_index:end_index]

        for client in list_clients_transformed:
            if i == 0: 
                client.heterogeneity_class = 'normal'
            elif i == 1:
                client.data['x'] = 255 - client.data['x']  # Invert greyscale
                client.data['x_test'] = 255 - client.data['x_test']  # Invert greyscale for test data
                client.heterogeneity_class = 'inverted_greyscale'
            elif i == 2:
                h, w = client.data['x'].shape[1:3]
                client.data['x'] = client.data['x'][:, h//4:3*h//4, w//4:3*w//4]  # Apply zoom
                client.data['x'] = np.pad(client.data['x'], ((0, 0), (7, 7), (7, 7)), mode='constant', constant_values=0)  # Add padding
                client.data['x_test'] = client.data['x_test'][:, h//4:3*h//4, w//4:3*w//4]  # Apply zoom for test data
                client.data['x_test'] = np.pad(client.data['x_test'], ((0, 0), (7, 7), (7, 7)), mode='constant', constant_values=0)  # Add padding for test data
                client.heterogeneity_class = 'zoom'
            elif i == 3:
                h, w = client.data['x'].shape[1:3]
                client.data['x'] = 255 - client.data['x']  # Invert greyscale
                client.data['x'] = client.data['x'][:, h//4:3*h//4, w//4:3*w//4]  # Apply zoom
                client.data['x'] = np.pad(client.data['x'], ((0, 0), (7, 7), (7, 7)), mode='constant', constant_values=0)  # Add padding
                client.data['x_test'] = 255 - client.data['x_test']  # Invert greyscale for test data
                client.data['x_test'] = client.data['x_test'][:, h//4:3*h//4, w//4:3*w//4]  # Apply zoom for test data
                client.data['x_test'] = np.pad(client.data['x_test'], ((0, 0), (7, 7), (7, 7)), mode='constant', constant_values=0)  # Add padding for test data
                client.heterogeneity_class = 'inverted_greyscale_zoom'
            
            data_preparation(client, row_exp)

        list_clients[start_index:end_index] = list_clients_transformed

    list_clients  = list_clients[:end_index]

    return list_clients

def apply_channel_invertion(list_clients: list, row_exp: dict) -> list:
    """ Utility function to apply normal, and 3 transformations of 2 channels swapping to 1/4 of Clients 

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients
    """
    
    number_of_client_per_transformation = row_exp['num_clients'] // 4

    for i in range(0, 4):
        
        start_index = i * number_of_client_per_transformation
        end_index = (i + 1) * number_of_client_per_transformation

        list_clients_transformed = list_clients[start_index:end_index]

        for client in list_clients_transformed:
            if i == 0: 
                client.heterogeneity_class = 'normal'
            elif i == 1:
                if client.data['x'].shape[-1] == 3:  # Check if the image has 3 channels
                    client.data['x'] = client.data['x'][:, :, :, [1, 0, 2]]  # Swap channel 0 and 1
                    client.data['x_test'] = client.data['x_test'][:, :, :, [1, 0, 2]]  # Swap channel 0 and 1 for test data
                    client.heterogeneity_class = 'swapped_channels_0_1'
            elif i == 2:
                if client.data['x'].shape[-1] == 3:  # Check if the image has 3 channels
                    client.data['x'] = client.data['x'][:, :, :, [2, 1, 0]]  # Swap channel 0 and 2
                    client.data['x_test'] = client.data['x_test'][:, :, :, [2, 1, 0]]  # Swap channel 0 and 2 for test data
                    client.heterogeneity_class = 'swapped_channels_0_2'
            elif i == 3:
                if client.data['x'].shape[-1] == 3:  # Check if the image has 3 channels
                    client.data['x'] = client.data['x'][:, :, :, [0, 2, 1]]  # Swap channel 1 and 2
                    client.data['x_test'] = client.data['x_test'][:, :, :, [0, 2, 1]]  # Swap channel 1 and 2 for test data
                    client.heterogeneity_class = 'swapped_channels_1_2'
            
            data_preparation(client, row_exp)

        list_clients[start_index:end_index] = list_clients_transformed

    list_clients = list_clients[:end_index]

    return list_clients

def apply_labels_skew(list_clients : list, row_exp : dict, list_ratios : list) -> list:
    """ 
    Utility function to apply label skew to Clients' data, ensuring equal distribution across heterogeneity classes.

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients with applied label skew
    """
    # Get unique heterogeneity classes
    heterogeneity_set = set([client.heterogeneity_class for client in list_clients])
    
    # Number of skews and clients per skew
    n_skews = len(list_ratios)
    
    # Number of clients per skew in each heterogeneity class
    clients_per_heterogeneity_class = row_exp['num_clients'] // len(heterogeneity_set)
    n_clients_by_skew = clients_per_heterogeneity_class // n_skews

    # Iterate over heterogeneity classes
    for heterogeneity_class in heterogeneity_set:
        # Filter clients by heterogeneity class
        clients_in_class = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
        
        # Apply skew to each set of clients in this heterogeneity class
        for i in range(n_skews):
            # Determine the indices for the clients in the current skew
            start_index = i * n_clients_by_skew
            end_index = (i + 1) * n_clients_by_skew

            # Get the clients that will receive the current skew
            list_clients_skewed = clients_in_class[start_index:end_index]

            # Apply the skew to each client
            for client in list_clients_skewed:
                unbalancing(client, list_ratios[i])
                data_preparation(client, row_exp)
                setattr(client, 'skew', f"lbl_skew_{str(i)}")

            # Update the clients list with the skewed clients
            for j, client in enumerate(list_clients_skewed):
                # Find the original index and replace the client
                original_index = list_clients.index(client)
                list_clients[original_index] = client
    
    return list_clients



def apply_quantity_skew(list_clients : list, row_exp : dict, list_skews : list,skew_type = 1 ) -> list:
    
    """ Utility function to apply quantity skew to Clients' data 
     For each element in list_skews, apply the skew to an equal subset of Clients 


    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
        list_skew : List of float 0 < i < 1  with quantity skews to subsample data
        type : Type of quantity skew
    
    Returns:
        Updated list of clients
    """
    
    n_max_samples = 100 # TODO: parameterize by dataset

    n_skews = len(list_skews)
    n_clients_by_skew = row_exp['num_clients'] // n_skews  

    dict_clients = [get_clients_data(n_clients_by_skew,
                                    int(n_max_samples * skew),
                                    row_exp['dataset'],
                                    row_exp['nn_model']) 
                                    for skew in list_skews] 
    dict_clients_test = get_clients_data(row_exp['num_clients'],
                                    n_max_samples,
                                    row_exp['dataset'],
                                    row_exp['nn_model'])
    list_clients = []
    if skew_type == 1 :
        for c in range(n_clients_by_skew):
            for s in range(len(list_skews)):
                client = Client(c * len(list_skews)+ s, dict_clients[s][c])
                setattr(client,"skew", "qt-skew_"+ str(list_skews[s]))
                list_clients.append(client)
    
    if skew_type == 2 :
        for s in range(len(list_skews)):
            for c in range(n_clients_by_skew):
                client = Client(c * len(list_skews)+ s, dict_clients[s][c])
                setattr(client,"skew", "qt-skew_"+ str(list_skews[s]))
                list_clients.append(client)

    for n_client, client in enumerate(list_clients):
        client.data['x_test'] = dict_clients_test[n_client]['x_test']
        client.data['y_test'] = dict_clients_test[n_client]['y_test']
        data_preparation(client, row_exp)

    return list_clients



def apply_features_skew(list_clients: list, row_exp: dict) -> list:
    """
    Utility function to apply feature skew to Clients' data.

    Arguments:
        list_clients: List of Client Objects with specific heterogeneity_class.
        row_exp: The current experiment's global parameters.
    
    Returns:
        Updated list of clients.
    """
    n_skew_types = 4  # Number of skew types (erosion, dilatation, none, big dilatation)
    n_clients = len(list_clients)
    n_clients_by_skew = n_clients // n_skew_types

    # Loop over skew types and apply corresponding transformations
    for i, skew_type in enumerate(['erosion', 'dilatation', 'none', 'big_dilatation']):
        start_index = i * n_clients_by_skew
        end_index = (i + 1) * n_clients_by_skew if i < n_skew_types - 1 else n_clients

        for client in list_clients[start_index:end_index]:
            if skew_type == 'erosion':
                client.data['x'] = erode_images(client.data['x'])
                client.data['x_test'] = erode_images(client.data['x_test'])
                client.heterogeneity_class = 'erosion'
            elif skew_type == 'dilatation':
                client.data['x'] = dilate_images(client.data['x'])
                client.data['x_test'] = dilate_images(client.data['x_test'])    
                client.heterogeneity_class = 'dilatation'
            elif skew_type == 'big_dilatation':
                client.data['x'] = dilate_images(client.data['x'], kernel_size=(8, 8))
                client.data['x_test'] = dilate_images(client.data['x_test'], kernel_size=(8, 8))
                client.heterogeneity_class = 'big_dilatation'
            else: 
                client.heterogeneity_class = 'none'
            
            # Prepare client data for further processing
            data_preparation(client, row_exp)

    return list_clients


def swap_labels(labels : list, client : Client, heterogeneity : int) -> Client:

    """ Utility Function for label swapping used for concept shift on labels. Sets the attribute "heterogeneity class"
    
    Arguments:
        labels : Labels to swap
        client : The Client object whose data we want to apply the swap on
    Returns:
        Client with labels swapped
    """

    newlabellist = client.data['y']
    newlabellist_test = client.data['y_test'] 

    otherlabelindex = newlabellist==labels[1]
    otherlabelindex_test = newlabellist_test==labels[1]
    
    newlabellist[newlabellist==labels[0]]=labels[1]
    newlabellist_test[newlabellist_test==labels[0]]=labels[1]
    
    newlabellist[otherlabelindex] = labels[0]
    newlabellist_test[otherlabelindex_test] = labels[0]
    
    client.data['y']= newlabellist
    client.data['y_test'] = newlabellist_test
    
    client.heterogeneity_class = heterogeneity
    return client

def global_test_data(list_clients: list, row_exp: dict) -> None:
    """Update the test and val loader to be global for each heterogeneity class

    Arguments:
        list_clients : The list of Client Objects
        row_exp : The current experiment's global parameters
        
    Returns:
        Train and test torch DataLoaders with data of all Clients
    """
    
    list_heterogeneities = list(dict.fromkeys([client.heterogeneity_class for client in list_clients]))
    
    for heterogeneity in list_heterogeneities:
        list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity]
        _, val_loader, test_loader = centralize_data(list_clients_filtered, row_exp)
        for client in list_clients:
            if client.heterogeneity_class == heterogeneity:
                client.data_loader['val'] = val_loader
                client.data_loader['test'] = test_loader

    return 

def centralize_data(list_clients: list, row_exp: dict) -> Tuple[DataLoader, DataLoader]:
    """Centralize data of the federated learning setup for central model comparison

    Arguments:
        list_clients : The list of Client Objects
        row_exp : The current experiment's global parameters

        
    Returns:
        Train and test torch DataLoaders with data of all Clients
    """
    
    from torchvision import transforms
    import torch 
    from torch.utils.data import DataLoader,TensorDataset
    import numpy as np 
    
# Define data augmentation transforms
    train_transform, val_transform, test_transform = data_transformation(row_exp)

    # Concatenate training data from all clients
    x_train = np.concatenate([list_clients[id].train_test['x_train'] for id in range(len(list_clients))], axis=0)
    y_train = np.concatenate([list_clients[id].train_test['y_train'] for id in range(len(list_clients))], axis=0)

    # Concatenate validation data from all clients
    x_val = np.concatenate([list_clients[id].train_test['x_val'] for id in range(len(list_clients))], axis=0)
    y_val = np.concatenate([list_clients[id].train_test['y_val'] for id in range(len(list_clients))], axis=0)

    # Concatenate test data from all clients
    x_test = np.concatenate([list_clients[id].train_test['x_test'] for id in range(len(list_clients))], axis=0)
    y_test = np.concatenate([list_clients[id].train_test['y_test'] for id in range(len(list_clients))], axis=0)

    # Create Custom Datasets
    train_dataset = CustomDataset(x_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(x_val, y_val, transform=val_transform)
    test_dataset = CustomDataset(x_test, y_test, transform=test_transform)
    # Create DataLoaders with different batch sizes for CIFAR-10
    batch_size = 256 if row_exp['dataset'] == 'cifar10' else 128
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader





def unbalancing(client: Client, ratio_list: list) -> Client:
    """ Downsample the dataset of a client using the ratios provided for each label.

    Arguments: 
        client : Client whose dataset we want to downsample
        ratio_list : Ratios to use for downsampling each label (from 0 to 9)
    """
    
    import pandas as pd
    from imblearn.datasets import make_imbalance
    from math import prod

    def ratio_func(y, multiplier, minority_class):
        from collections import Counter
        target_stats = Counter(y)
        return {minority_class: int(multiplier * target_stats[minority_class])}

    x_train = client.data['x']
    y_train = client.data['y']
    
    orig_shape = x_train.shape
    
    # Flatten the images
    X_resampled = x_train.reshape(-1, prod(orig_shape[1:]))
    y_resampled = y_train
    
    # Ensure ratio_list has exactly 10 elements (for labels 0 to 9)
    if len(ratio_list) != 10:
        raise ValueError("The ratio_list must have exactly 10 elements, one for each label.")
    
    # Loop over labels from 0 to 9
    for i in range(10):  # Iterating through each label from 0 to 9
        X = pd.DataFrame(X_resampled)
        
        X_resampled, y_resampled = make_imbalance(
            X,
            y_resampled,
            sampling_strategy=ratio_func,
            **{"multiplier": ratio_list[i], "minority_class": i}
        )

    client.data['x'] = X_resampled.to_numpy().reshape(-1, *orig_shape[1:])
    client.data['y'] = y_resampled
    
    return client

def dilate_images(x : ndarray, kernel_size : tuple = (3, 3)) -> ndarray:
    
    """ Perform dilation operation on a batch of images using a given kernel.
    Make image 'bolder' for features distribution skew setup
    
    
    Arguments:
        x : Input batch of images (3D array with shape (n, height, width)).
        kernel_size : Size of the structuring element/kernel for dilation.

    Returns:
        ndarray Dilation results for all images in the batch.
    """
    
    import cv2
    import numpy as np 

    n = x.shape[0]

    dilated_images = np.zeros_like(x, dtype=np.uint8)
    
    # Create the kernel for dilation
    kernel = np.ones(kernel_size, np.uint8)

    for i in range(n):
    
        dilated_image = cv2.dilate(x[i], kernel, iterations=1)
    
        dilated_images[i] = dilated_image

    return dilated_images


def erode_images(x : ndarray, kernel_size : tuple =(3, 3)) -> ndarray:
    """
    Perform erosion operation on a batch of images using a given kernel.
    Make image 'finner' for features distribution skew setup

    Arguments:
        x : Input batch of images (3D array with shape (n, height, width)).
        kernel_size :  Size of the structuring element/kernel for erosion.

    Returns:
        ndarray of Erosion results for all images in the batch.
    """
    
    import cv2
    import numpy as np 

    n = x.shape[0]  
    eroded_images = np.zeros_like(x, dtype=np.uint8)

    # Create the kernel for erosion
    kernel = np.ones(kernel_size, np.uint8)

    # Iterate over each image in the batch
    for i in range(n):
        # Perform erosion on the current image
        eroded_image = cv2.erode(x[i], kernel, iterations=1)
        # Store the eroded image in the results array
        eroded_images[i] = eroded_image

    return eroded_images


def get_uid(str_obj: str) -> str:
    """
    Generates an (almost) unique Identifier given a string object.
    Note: Collision probability is low enough to be functional for the use case desired which is to uniquely identify experiment parameters using an int
    """

    import hashlib
    hash = hashlib.sha1(str_obj.encode("UTF-8")).hexdigest()
    return hash

    
