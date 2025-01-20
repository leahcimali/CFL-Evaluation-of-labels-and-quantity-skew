import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import pandas as pd

from src.models import ImageClassificationBase
from src.fedclass import Server


def ColdStart(my_server : Server, list_clients : list, row_exp : dict, algorithm : str = 'kmeans', clustering_metric : str ='euclidean',ponderated : bool = True, alpha :int =6)-> None:
    """ Cold start clustering for iterative server-side CFL. Create first clustering iteration using a subsample of all clients. 
    Credit -> Inspired by https://github.com/morningD/FlexCFL
    
    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
        algorithm : Clustering algorithm used on server can be kmeans or agglomerative clustering
        clustering_metric : Euclidean, cosine or MADC
        alpha : Parameter for client selection
        coldstart : If the cold start algorithm is done 
    """
    import random
    from src.utils_fed import k_means_clustering, Agglomerative_Clustering, fedavg, send_clusters_models_to_clients,client_migration
    import copy
    # select a subsample of clients for coldstart
    selected_clients = random.sample(list_clients, k=min(row_exp['num_clusters']*alpha, len(list_clients)))
    #unselected_clients = [client for client in list_clients if client not in selected_clients]
    send_clusters_models_to_clients(selected_clients, my_server)
    for client in selected_clients :
        client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
        if algorithm != 'kmeans' :
            Agglomerative_Clustering(my_server,selected_clients, row_exp['num_clusters'], clustering_metric, row_exp['seed'])
        else: 
            k_means_clustering(my_server,selected_clients, row_exp['num_clusters'], row_exp['seed'],metric= clustering_metric)
    fedavg(my_server, selected_clients,ponderated= ponderated)
    return

def FedGroup(my_server : Server, list_clients : list, row_exp : dict, algorithm : str = 'kmeans', clustering_metric : str ='euclidean',ponderated :bool = True, alpha :int =6)-> None:
    """ FedGroup for one communication round
    Credit -> Inspired by https://github.com/morningD/FlexCFL
    
    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
        algorithm : Clustering algorithm used on server can be kmeans or agglomerative clustering
        clustering_metric : Euclidean, cosine or MADC
        alpha : Parameter for client selection
        coldstart : If the cold start algorithm is done 
    """
    import random
    from src.utils_fed import fedavg, send_clusters_models_to_clients,client_migration
    import copy
    # Cold_start
    setattr(my_server, 'num_clusters', row_exp['num_clusters'])
    my_server.clusters_models= {cluster_id: copy.deepcopy(my_server.model) for cluster_id in range(row_exp['num_clusters'])}  
    ColdStart(my_server,list_clients,row_exp,algorithm,clustering_metric)
    
    round_counter = 0
    while round_counter < row_exp['federated_rounds'] or len([client for client in list_clients if client.cluster_id is None]) > 0:
        selected_clients = random.sample(list_clients, k=min(row_exp['num_clusters']*alpha, len(list_clients)))
        send_clusters_models_to_clients(selected_clients, my_server)
        
        for client in selected_clients:
            client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
            if client.cluster_id is None:
                client_migration(my_server, client)

        fedavg(my_server, selected_clients,ponderated=ponderated)

        # Update round counter only if within federated rounds
        if round_counter < row_exp['federated_rounds']:
            round_counter += 1
    return

def run_cfl_server_side(my_server : Server, list_clients : list, row_exp : dict, algorithm : str = 'kmeans', clustering_metric : str ='euclidean',iterative = False,ponderated = True) -> pd.DataFrame:
    
    """ Driver function for server-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights .
    
    Arguments:

        my_server : Main server model    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
        algorithm : Clustering algorithm used on server can be kmeans or agglomerative clustering
        clustering_metric : euclidean, cosine or MADC

    Returns:

        df_results : dataframe with the experiment results
    """

    from src.utils_fed import k_means_clustering, Agglomerative_Clustering
    import copy
    import torch 

    torch.manual_seed(row_exp['seed'])
    
    if iterative == True :
        FedGroup(my_server,list_clients,row_exp,algorithm,clustering_metric,ponderated)      
    else:
        cold_start = row_exp
        cold_start['federated_rounds'] = 2 
        my_server = train_federated(my_server, list_clients, cold_start, use_clusters_models = False,ponderated=ponderated)
        my_server.clusters_models= {cluster_id: copy.deepcopy(my_server.model) for cluster_id in range(row_exp['num_clusters'])}  
        setattr(my_server, 'num_clusters', row_exp['num_clusters'])
        
        if algorithm == 'cheat':
            # Use Personalized for Clustered Federated Learning with knowledge of client heterogeneity class
            # Used as a benchmark for CFL.
            print('Using personalized Federated Learning!')
            heterogeneity_classes = set([client.heterogeneity_class for client in list_clients])
            cluster_mapping = {cls: idx for idx, cls in enumerate(heterogeneity_classes)}
            for client in list_clients:
                client.cluster_id = cluster_mapping[client.heterogeneity_class]        

        elif algorithm == 'agglomerative' :
            Agglomerative_Clustering(my_server,list_clients, row_exp['num_clusters'], clustering_metric, row_exp['seed'])
        
        elif algorithm == 'kmeans': 
            k_means_clustering(my_server,list_clients, row_exp['num_clusters'], row_exp['seed'])

        my_server = train_federated(my_server, list_clients, row_exp, use_clusters_models = True)

    for client in list_clients :

        acc = test_model(my_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

    return df_results 


def run_cfl_client_side(my_server : Server, list_clients : list, row_exp : dict,ponderated : bool = True) -> pd.DataFrame:

    """ Driver function for client-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights (k-means).
    

    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
    """

    from src.utils_fed import  set_client_cluster, fedavg
    import torch

    torch.manual_seed(row_exp['seed'])
    
    for _ in range(row_exp['federated_rounds']):

        for client in list_clients:

            client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)

        fedavg(my_server, list_clients,ponderated)

        set_client_cluster(my_server, list_clients, row_exp)

    for client in list_clients :

        acc = test_model(my_server.clusters_models[client.cluster_id], client.data_loader['test'])
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results

def run_cfl_hybrid(my_server : Server, list_clients : list, row_exp : dict, algorithm : str = 'kmeans', clustering_metric : str ='euclidean',ponderated : bool = False) -> pd.DataFrame:
    
    """ Driver function for server-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights .
    
    Arguments:

        my_server : Main server model    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
        algorithm : Clustering algorithm used on server can be kmeans or agglomerative clustering
        clustering_metric : euclidean, cosine or MADC

    Returns:

        df_results : dataframe with the experiment results
    """

    from src.utils_fed import k_means_clustering, Agglomerative_Clustering, fedavg, set_client_cluster
    import copy
    import torch 

    torch.manual_seed(row_exp['seed'])
    
    
    cold_start = row_exp
    cold_start['federated_rounds'] = 1
    my_server = train_federated(my_server, list_clients, cold_start, use_clusters_models = False,ponderated=False)
    my_server.clusters_models= {cluster_id: copy.deepcopy(my_server.model) for cluster_id in range(row_exp['num_clusters'])}  
    setattr(my_server, 'num_clusters', row_exp['num_clusters'])
    for client in list_clients:
        client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
    for round in range(row_exp['federated_rounds']):
        if algorithm == 'agglomerative' :
            Agglomerative_Clustering(my_server,list_clients, row_exp['num_clusters'], clustering_metric, row_exp['seed'])
            
        elif algorithm == 'kmeans': 
            k_means_clustering(my_server,list_clients, row_exp['num_clusters'], row_exp['seed'])
        fedavg(my_server, list_clients)
        set_client_cluster(my_server, list_clients, row_exp)
        for client in list_clients:
            client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
    for round in range(row_exp['federated_rounds']):
        fedavg(my_server, list_clients)
        set_client_cluster(my_server, list_clients, row_exp)
        if round != row_exp['federated_rounds']//2 -1 :
            for client in list_clients:
                client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
    for client in list_clients :

        acc = test_model(my_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

    return df_results 

def run_benchmark(model_server : Server, list_clients : list, row_exp : dict) -> pd.DataFrame:

    """ Benchmark function to calculate baseline FL results and ``optimal'' personalization results if clusters are known in advance

    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
    """

    import pandas as pd 
    import torch
    import copy

    from src.utils_data import centralize_data

    list_heterogeneities = list(dict.fromkeys([client.heterogeneity_class for client in list_clients]))
    
    torch.manual_seed(row_exp['seed'])
    torch.use_deterministic_algorithms(True)

    curr_model = model_server.model

    if row_exp['exp_type'] == 'pers-centralized':
        curr_model = model_server.model
        for heterogeneity_class in list_heterogeneities:
            list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
            train_loader, val_loader, test_loader = centralize_data(list_clients_filtered,row_exp)
            model_trained, _ = train_central(curr_model, train_loader, val_loader, row_exp) 

            global_acc = test_model(model_trained, test_loader) 
                    
            for client in list_clients_filtered : 
    
                setattr(client, 'accuracy', global_acc)
    
    elif row_exp['exp_type'] == 'global-federated':
                
        model_trained = train_federated(model_server, list_clients, row_exp, use_clusters_models = False)

        _, _,test_loader = centralize_data(list_clients,row_exp)
        global_acc = test_model(model_trained.model, test_loader) 
                    
        for client in list_clients : 
    
            setattr(client, 'accuracy', global_acc)
    elif row_exp['exp_type'].split('-')[0] == 'fedprox':
        if len(row_exp['exp_type'].split('-')) == 2:
            mu = row_exp['exp_type'].split('-')[1]
            row_exp['exp_type'] = 'fedprox'
        else :
            mu =0.01
        model_trained = train_federated(model_server, list_clients, row_exp, use_clusters_models = False, fedprox=mu)

        _, _,test_loader = centralize_data(list_clients,row_exp)
        global_acc = test_model(model_trained.model, test_loader) 
                    
        for client in list_clients : 
    
            setattr(client, 'accuracy', global_acc)
        
    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results


def train_federated(main_model, list_clients, row_exp, use_clusters_models = False, ponderated = True,fedprox = False):
    
    """Controler function to launch federated learning

    Arguments:

        main_model: Server model used in our experiment
        list_clients: A list of Client Objects used as nodes in the FL protocol  
        row_exp: The current experiment's global parameters
        use_clusters_models: Boolean to determine whether to use personalization by clustering
        fedprox : Boolean to determine whether to use fedprox 
    """
    
    from src.utils_fed import send_server_model_to_client, send_clusters_models_to_clients, fedavg
    if fedprox :
        mu =fedprox
    else : 
        mu =0
    for i in range(0, row_exp['federated_rounds']):

        accs = []

        if use_clusters_models == False:
        
            send_server_model_to_client(list_clients, main_model)

        else:

            send_clusters_models_to_clients(list_clients, main_model)

        for client in list_clients:
            print(f"Training client {client.id} with dataset of size {client.data['x'].shape}")
            client.model, curr_acc = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp, mu)
            accs.append(curr_acc)
            fedavg(main_model, list_clients,ponderated)

    return main_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def evaluate(model : nn.Module, val_loader : DataLoader) -> dict:
    
    """ Returns a dict with loss and accuracy information"""
    model.to(device)
    model.eval()
    outputs =[]
    for batch in val_loader:
        # Move entire batch to the correct device
        batch = [item.to(device) for item in batch]
        
        # Call the validation step and append to outputs
        output = model.validation_step(batch,device)
        outputs.append(output)
    return model.validation_epoch_end(outputs)
'''
def train_central(model: ImageClassificationBase, train_loader: DataLoader, val_loader: DataLoader, row_exp: dict):
    """ Main training function for centralized learning
    
    Arguments:
        model : Server model used in our experiment
        train_loader : DataLoader with the training dataset
        val_loader : Dataloader with the validation dataset
        row_exp : The current experiment's global parameters

    Returns:
        (model, history) : base model with trained weights / results at each training step
    """

    # Check if CUDA is available and set the device
    
    # Move the model to the appropriate device
    model.to(device)

    opt_func = torch.optim.Adam  # if row_exp['nn_model'] == "linear" else torch.optim.Adam
    lr = 0.001
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(row_exp['centralized_epochs']):
        
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Move batch to the same device as the model
            inputs, labels = [item.to(device) for item in batch]
    
            # Pass the unpacked inputs and labels to the model's training step
            loss = model.training_step((inputs, labels),device)            
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
                
        result = evaluate(model, val_loader)  # Ensure evaluate handles CUDA as needed
        result['train_loss'] = torch.stack(train_losses).mean().item()        
        
        model.epoch_end(epoch, result)
        
        history.append(result)
    
    return model, history
'''
def train_central(model: ImageClassificationBase, train_loader: DataLoader, val_loader: DataLoader, 
    row_exp: dict, mu: float = 0.0):

    """
    Main training function for centralized learning with optional FedProx regularization.
    
    Arguments:
        model : Local model to be trained
        train_loader : DataLoader with the training dataset
        val_loader : DataLoader with the validation dataset
        row_exp : Experiment's global parameters
        my_server : Instance of the Server class for FedProx regularization
        mu : Regularization coefficient for FedProx (default: 0.0, ignored if 0)

    Returns:
        (model, history) : Trained model with updated weights and training history
    """
    import copy

    # Move the model and server's model to the device if necessary
    server_model = copy.deepcopy(model)

    if mu > 0 :
        server_model.to(device)

    # Optimizer setup
    opt_func = torch.optim.Adam
    lr = 0.001
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(row_exp['centralized_epochs']):
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move batch to the same device as the model
            inputs, labels = [item.to(device) for item in batch]

            # Forward pass and calculate base loss
            loss = model.training_step((inputs, labels), device)
            
            # If mu > 0, apply FedProx regularization
            if mu > 0 :
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), server_model.parameters()):
                    proximal_term += (w - w_t).norm(2) ** 2
                loss += (mu / 2) * proximal_term  # Add FedProx term

            # Backward pass and optimization step
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation step
        result = evaluate(model, val_loader)  # Ensure evaluate handles CUDA as needed
        result['train_loss'] = torch.stack(train_losses).mean().item()

        # Print epoch results and add to history
        model.epoch_end(epoch, result)
        history.append(result)

    return model, history
    

def test_model(model: nn.Module, test_loader: DataLoader) -> float:
    """ Calculates model accuracy (percentage) on the <test_loader> Dataset
    
    Arguments:
        model : the input server model
        test_loader : DataLoader with the dataset to use for testing
    """
    
    criterion = nn.CrossEntropyLoss()

    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the device
    model.to(device)

    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # No need to track gradients in evaluation

        for batch in test_loader:
            inputs, labels = [item.to(device) for item in batch]
            
            labels = labels.long()        
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Calculate total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average test loss over all examples
    test_loss = test_loss / len(test_loader.dataset)

    # Calculate accuracy percentage
    accuracy = (correct / total) * 100

    return accuracy
