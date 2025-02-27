import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import pandas as pd

from src.models import ImageClassificationBase
from src.fedclass import Server

    
        

def ColdStart(fl_server : Server, list_clients : list, row_exp : dict, algorithm : str = 'kmeans', clustering_metric : str ='euclidean',
              ponderated : bool = True, client_sampling = 0.25)-> None:
    """ Cold start clustering for iterative server-side CFL. Create first clustering iteration using a subsample of all clients. 
    Credit -> Inspired by https://github.com/morningD/FlexCFL
    
    Arguments:

        fl_server : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
        algorithm : Clustering algorithm used on server can be kmeans or agglomerative clustering
        clustering_metric : Euclidean, cosine or MADC
        client_sampling : Parameter for client selection
        coldstart : If the cold start algorithm is done 
    """
    import random
    from src.utils_fed import k_means_clustering, Agglomerative_Clustering, fedavg, send_clusters_models_to_clients,client_migration
    import copy
    # select a subsample of clients for coldstart
    
    selected_clients = random.sample(list_clients, k=int(client_sampling*len(list_clients)))
    send_clusters_models_to_clients(selected_clients, fl_server)
    for client in selected_clients :
        print("Training client ", client.id)
        client.model, _, acc , client.update = train_model(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
        client.round_acc.append(acc)
        if clustering_metric == 'madc' :
            Agglomerative_Clustering(fl_server,selected_clients, row_exp['num_clusters'], clustering_metric, row_exp['seed'],linkage_type='complete')
        else: 
            k_means_clustering(fl_server,selected_clients, row_exp['num_clusters'], row_exp['seed'], metric= clustering_metric)
    fedavg(fl_server, selected_clients,ponderated= ponderated)
    return

def FedGroup(fl_server : Server, list_clients : list, row_exp : dict, alpha, algorithm : str = 'kmeans', clustering_metric : str ='euclidean',ponderated :bool = True)-> pd.DataFrame:
    """ FedGroup for one communication round
    Credit -> Inspired by https://github.com/morningD/FlexCFL
    
    Arguments:

        fl_server : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
        algorithm : Clustering algorithm used on server can be kmeans or agglomerative clustering
        clustering_metric : Euclidean, cosine or MADC
        alpha : Parameter for client selection
        coldstart : If the cold start algorithm is done 
     Returns:

        df_results : dataframe with the experiment results
    """
    import random
    from src.utils_fed import fedavg, send_clusters_models_to_clients,client_migration
    import copy
    # Cold_start
    # Because of client selection we need to multiply the number of rounds by 4 to compare with other algorithms. 

    setattr(fl_server, 'num_clusters', row_exp['num_clusters'])
    fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}  
    ColdStart(fl_server,list_clients,row_exp,algorithm,clustering_metric)
    
    round_counter = 0
    while round_counter < row_exp['rounds'] or len([client for client in list_clients if client.cluster_id is None]) > 0:
        selected_clients = random.sample(list_clients, k=min(row_exp['num_clusters']*alpha, len(list_clients)))
        send_clusters_models_to_clients(selected_clients, fl_server)
        
        for client in selected_clients:
            print("Training client ", client.id)
            client.model, _ , acc, client.update= train_model(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
            client.round_acc.append(acc)

            if client.cluster_id is None:
                client_migration(fl_server, client)

        fedavg(fl_server, selected_clients,ponderated=ponderated)

        # Update round counter only if within federated rounds
        if round_counter < row_exp['rounds']:
            round_counter += 1
    for client in list_clients :

        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

    return df_results 

    
def run_cfl_server_side(fl_server : Server, list_clients : list, row_exp : dict, ponderated = True) -> pd.DataFrame:
    
    """ Driver function for server-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights .
    
    Arguments:

        fl_server : Main server model    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters

    Returns:

        df_results : dataframe with the experiment results
    """

    from src.utils_fed import k_means_clustering, Agglomerative_Clustering
    import copy
    import torch 

    torch.manual_seed(row_exp['seed'])
    
    # Half the rounds before clustering and half after

    row_exp['rounds'] =  row_exp['rounds']//2
    
    fl_server = train_federated(fl_server, list_clients, row_exp, use_clusters_models = False, ponderated=ponderated)
    fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}  
    setattr(fl_server, 'num_clusters', row_exp['num_clusters'])
        
    if row_exp['exp_type'] == 'oracle-cfl':
        # Use Personalized for Clustered Federated Learning with knowledge of client heterogeneity class
        # Used as a benchmark for CFL.
        print('Using personalized Federated Learning!')
        heterogeneity_classes = set([client.heterogeneity_class for client in list_clients])
        cluster_mapping = {cls: idx for idx, cls in enumerate(heterogeneity_classes)}
        for client in list_clients:
            client.cluster_id = cluster_mapping[client.heterogeneity_class]        

    elif row_exp['exp_type'] == 'hcfl' :
        Agglomerative_Clustering(fl_server,list_clients, row_exp['num_clusters'], row_exp['params'], row_exp['seed'])
    
    elif row_exp['exp_type'] == 'cfl': 
        k_means_clustering(fl_server,list_clients, row_exp['num_clusters'], row_exp['seed'])

    fl_server = train_federated(fl_server, list_clients, row_exp, use_clusters_models = True)

    for client in list_clients :

        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

    return df_results 


def run_cfl_IFCA(fl_server : Server, list_clients : list, row_exp : dict, ponderated : bool = True) -> pd.DataFrame:

    """ Driver function for client-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights (k-means).
    

    Arguments:

        fl_server : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
    """

    from src.utils_fed import  set_client_cluster, fedavg
    import torch

    torch.manual_seed(row_exp['seed'])

    for _ in range(row_exp['rounds']):

        for client in list_clients:
            print("Training client ", client.id)
            client.model, _, acc, client.update = train_model(client.model.to(device), client.data_loader['train'], client.data_loader['val'], row_exp)
            client.round_acc.append(acc)

        fedavg(fl_server, list_clients,ponderated)

        set_client_cluster(fl_server, list_clients, row_exp)
    
    list_clients_dict = [] 
    for client in list_clients : 
        validation = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['val'])
        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])
        setattr(client, 'accuracy', acc)
        client_dict = client.to_dict()
        client_dict['validation'] = validation
        list_clients_dict.append(client_dict)
        
    df_results = pd.DataFrame.from_records(list_clients_dict)
    return df_results

def run_cfl_cornflqs(fl_server : Server, list_clients : list, row_exp : dict, algorithm : str = 'kmeans', clustering_metric : str ='euclidean', ponderated : bool = False) -> pd.DataFrame:
    
    """ Driver function for server-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights .
    
    Arguments:

        fl_server : Main server model    
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
    import ast
    torch.manual_seed(row_exp['seed'])
    try:
        cold_rounds, server_client_rounds, client_only_rounds = ast.literal_eval(row_exp['params'])
    except Exception as e:
        print('Error in params:', e, '!')
        print('Using defaults parameters')
        cold_rounds, server_client_rounds, client_only_rounds = 1, row_exp['rounds'] // 2, row_exp['rounds'] // 2
            
    row_exp['rounds'] = cold_rounds
    # Cold start
    # Train the federated model with unponderated fedavg for n = cold_rounds rounds
    fl_server = train_federated(fl_server, list_clients, row_exp, use_clusters_models = False, ponderated=False)
    fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}
    
    setattr(fl_server, 'num_clusters', row_exp['num_clusters'])
    for client in list_clients:
        print("Training client ", client.id)
        client.model, _ , acc, client.update = train_model(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
        client.round_acc.append(acc)
    # Server/Client aggrement rounds
    for round in range(server_client_rounds):
        if round == 0 :
            model_update = True
        else :
            model_update = False
        
        if algorithm == 'agglomerative' :
            Agglomerative_Clustering(fl_server,list_clients, row_exp['num_clusters'], clustering_metric, row_exp['seed'],model_update=model_update)

        elif algorithm == 'kmeans': 
            k_means_clustering(fl_server,list_clients, row_exp['num_clusters'], row_exp['seed'],clustering_metric,model_update)
        fedavg(fl_server, list_clients)
        
        set_client_cluster(fl_server, list_clients, row_exp)
        for client in list_clients:
            print("Training client ", client.id)
            client.model, _ , acc, client.update= train_model(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
            client.round_acc.append(acc)
    # Client only rounds
    for round in range(client_only_rounds):
        fedavg(fl_server, list_clients) # Do fedavg by cluster
        set_client_cluster(fl_server, list_clients, row_exp)
        for client in list_clients:
            print("Training client ", client.id)
            client.model, _ , acc, client.update= train_model(client.model, client.data_loader['train'], client.data_loader['val'], row_exp, mu = 0.1)
            client.round_acc.append(acc)

    for client in list_clients :

        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

    return df_results 

def run_benchmark(fl_server : Server, list_clients : list, row_exp : dict) -> pd.DataFrame:

    """ Benchmark function to calculate baseline FL results and ``optimal'' personalization results if clusters are known in advance

    Arguments:

        fl_server : Type of Server model needed    
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

    curr_model = fl_server.model

    if row_exp['exp_type'] == 'oracle-centralized':
        curr_model = fl_server.model
        # Concept shift on labels have contradictory labels, so they need to be separated.
        if row_exp['heterogeneity_type'] == 'concept-shift-on-labels':
            
            for heterogeneity_class in list_heterogeneities:
                list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
                train_loader, val_loader, test_loader = centralize_data(list_clients_filtered,row_exp)
                model_trained, _, _, _ = train_model(curr_model, train_loader, val_loader, row_exp) 

                global_acc = test_model(model_trained, test_loader) 
                        
                for client in list_clients_filtered : 

                    setattr(client, 'accuracy', global_acc)
        else :
            train_loader, val_loader, test_loader = centralize_data(list_clients,row_exp)
            model_trained, _, _, _ = train_model(curr_model, train_loader, val_loader, row_exp) 

            global_acc = test_model(model_trained, test_loader) 
                        
            for client in list_clients : 

                setattr(client, 'accuracy', global_acc)
    
    elif row_exp['exp_type'] == "fedavg":
                
        model_trained = train_federated(fl_server, list_clients, row_exp, use_clusters_models = False)

        _, _,test_loader = centralize_data(list_clients,row_exp)
        global_acc = test_model(model_trained.model, test_loader) 
                    
        for client in list_clients : 
    
            setattr(client, 'accuracy', global_acc)

    elif row_exp['exp_type'] == 'fedprox':
        try:
            mu = float(row_exp['params'])  
        except (TypeError, ValueError):  
            mu = 0.01  
        
        model_trained = train_federated(fl_server, list_clients, row_exp, use_clusters_models = False, fedprox_mu=mu)

        _, _,test_loader = centralize_data(list_clients,row_exp)
        global_acc = test_model(model_trained.model, test_loader) 
                    
        for client in list_clients : 
    
            setattr(client, 'accuracy', global_acc)
        
    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results


def train_federated(fl_server, list_clients, row_exp, use_clusters_models = False, ponderated = True, fedprox_mu : float = 0):
    
    """Controler function to launch federated learning

    Arguments:

        fl_server: Server model used in our experiment
        list_clients: A list of Client Objects used as nodes in the FL protocol  
        row_exp: The current experiment's global parameters
        use_clusters_models: Boolean to determine whether to use personalization by clustering
        fedprox_mu : value of mu for fedprox, if 0 do standard FedAVG
    """
    
    from src.utils_fed import send_server_model_to_client, send_clusters_models_to_clients, fedavg
    
    for i in range(int(row_exp['rounds'])):

        accs = []

        if use_clusters_models == False:
        
            send_server_model_to_client(list_clients, fl_server)

        else:

            send_clusters_models_to_clients(list_clients, fl_server)

        for client in list_clients:
            print(f"Training client {client.id} with dataset of size {client.data['x'].shape}")
            client.model, curr_acc, val_acc, client.update = train_model(client.model, client.data_loader['train'], client.data_loader['val'], row_exp, fedprox_mu)
            client.round_acc.append(val_acc)
            accs.append(curr_acc)
            fedavg(fl_server, list_clients,ponderated)

    return fl_server


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



def train_model(model: ImageClassificationBase, train_loader: DataLoader, val_loader: DataLoader, 
                  row_exp: dict, mu: float = 0.0, lr = 0.001, opt_func = torch.optim.Adam):
    """
    Main training function for local or centralized learning with optional FedProx regularization.
    
    Arguments:
        model : Local model to be trained
        train_loader : DataLoader with the training dataset
        val_loader : DataLoader with the validation dataset
        row_exp : Experiment's global parameters
        mu : Regularization coefficient for FedProx (default: 0.0, ignored if 0)

    Returns:
        (model, history, final_val_acc, weight_update) : Trained model, final train loss of the trained model, 
        validation set accuracy of the trained model, and the weights update of the model
    """
    import copy

    # Move the model to the device
    server_model = copy.deepcopy(model).to(device)

    # Optimizer setup
    history = []
    optimizer = opt_func(model.parameters(), lr)

    # Initialize variable to accumulate gradients
    avg_grad = [torch.zeros_like(param) for param in model.parameters()]

    for epoch in range(row_exp['epochs']):
        model.train()
        train_losses = []
        num_batches = 0

        for batch in train_loader:
            num_batches += 1
            # Move batch to the same device as the model
            inputs, labels = [item.to(device) for item in batch]

            # Forward pass and calculate base loss
            loss = model.training_step((inputs, labels), device)
            
            # If mu > 0, apply FedProx regularization
            if mu > 0:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), server_model.parameters()):
                    proximal_term += (w - w_t).norm(2) ** 2
                loss += (mu / 2) * proximal_term  # Add FedProx term

            # Backward pass
            train_losses.append(loss)
            loss.backward()

            # Accumulate gradients
            for i, param in enumerate(model.parameters()):
                if param.grad is None:  # **Change: Check for None gradients**
                    print(f"Gradient is None for parameter {i}")
                else:
                    avg_grad[i] += param.grad.clone().detach()  # **Change: Ensure gradients are tensors**

            # Optimization step
            optimizer.step()
            optimizer.zero_grad()
        
        # Normalize gradients to compute the average
        if num_batches > 0:  # **Change: Check if num_batches > 0 to avoid division by zero**
            for i in range(len(avg_grad)):
                avg_grad[i] /= num_batches

        # Validation step
        result = evaluate(model, val_loader)  # Ensure evaluate handles CUDA as needed
        result['train_loss'] = torch.stack(train_losses).mean().item()

        # Print epoch results and add to history
        #model.epoch_end(epoch, result)
        history.append(result)

    # Final validation accuracy
    val_acc = test_model(model, val_loader)
    train_loss = torch.stack(train_losses).mean().item()
    
    weight_update = {}
    for name, param in model.named_parameters():
        if name in server_model.state_dict():
            weight_update[name] = param.data - server_model.state_dict()[name]
    return model, train_loss, val_acc, weight_update
    

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

import networkx as nx
import numpy as np
from src.fedclass import Server, Client
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy

from src.metrics import cross_cluster_distance_client_to_cluster, cross_cluster_distance_cluster_to_cluster, compute_distance_matrix

def srfca(fl_server : Server, list_clients : list, row_exp : dict) -> pd.DataFrame:
  """
    Perform the SRFCA (Self-Refining Federated Clustering Algorithm) for federated learning.

    Arguments:
        fl_server (Server): The federated learning server that holds the global model and cluster models.
        list_clients (list): A list of client objects participating in the federated learning process.
        row_exp (dict): A dictionary containing experiment parameters including number of federated rounds and clustering settings.

    Returns:
        pd.DataFrame: A DataFrame containing the results (such as accuracy) for each client after federated rounds.
  """
  import ast
  from src.utils_training import test_model
  from src.utils_fed import send_clusters_models_to_clients
  from src.utils_training import train_model
  #SRFCA hyperparameters
  
  # ONE_SHOT STEP
  send_clusters_models_to_clients(list_clients,fl_server)

  # First Training
  for client in list_clients :
      
    print("Training client ", client.id)
    client.model, _, acc , client.update = train_model(client.model, client.data_loader['train'], client.data_loader['val'],row_exp)
    client.round_acc.append(acc)
  
  # Distance Matrix for ONE-SHOT Step
  similarity_matrix = compute_distance_matrix(list_clients)
  print(similarity_matrix)
  
  # Flatten the matrix and remove diagonal elements
  similarity_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]  
  percentile = 25
  lambda_threshold = np.percentile(similarity_values,percentile)
  beta = 0.15
  connection_size_t = 2 
  print('Distance Threshold : ', lambda_threshold)
  
  row_exp['num_clusters'] = one_shot(fl_server,list_clients,lambda_threshold,connection_size_t,similarity_matrix)
  while percentile > 10 :
    percentile = percentile - 3
    lambda_threshold = np.percentile(similarity_values, percentile)+ 0.0001
    if one_shot(fl_server,list_clients,lambda_threshold,connection_size_t,similarity_matrix) > row_exp['num_clusters'] :
        print('Updated Distance Threshold : ', lambda_threshold)
        row_exp['num_clusters'] = one_shot(fl_server,list_clients,lambda_threshold,connection_size_t,similarity_matrix)
        break
    
    
    row_exp['num_clusters'] = one_shot(fl_server,list_clients,lambda_threshold,connection_size_t,similarity_matrix)

  fl_server.num_clusters = row_exp['num_clusters']
  fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}  
  print('Initialized Clusters : '+ str(fl_server.num_clusters))

  for round in range(row_exp['rounds']):
    print('Communication Round ' + str(round+1))
    # REFINE STEP
    print('Doing Refine step')
    refine_step = True
    if round > 3 :
       refine_step = False
    refine(fl_server,list_clients,row_exp,beta,lambda_threshold,connection_size_t,refine_step)

  for client in list_clients :
        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

  df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

  return df_results 

def refine(fl_server : Server, list_clients : list, row_exp : dict,beta :float, lambda_threshold : float,connection_size_t : int,refine_step = False)-> None:
  """
    Refine the model aggregation and cluster assignments for clients by performing 
    trimmed mean aggregation and optionally reclustering and merging.

    Arguments:
        fl_server (Server): The federated learning server containing the global and cluster models.
        list_clients (list): A list of client objects involved in federated learning.
        row_exp (dict): A dictionary containing experiment parameters, including the number of clusters and federated rounds.
        beta (float): The fraction of extreme values to trim during the aggregation process.
        lambda_threshold (float): The threshold for cluster distance to determine if clusters should merge.
        connection_size_t (int): The minimum size of connected components to consider during merging.
        refine_step (bool, optional): Whether to perform reclustering and merging after aggregation. Default is False.

    Returns:
        None: This function modifies the server model and client assignments in place.
    """
  
  #STEP 1) Trimmed Mean on each cluster 
  print('trimmed Mean Step')
  for cluster_id in range(row_exp['num_clusters']) :
    cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id] 
    fl_server.clusters_models[cluster_id] = trimmed_mean_beta_aggregation(fl_server.clusters_models[cluster_id], cluster_clients_list,row_exp,beta)
  
  if refine_step == True :

    #STEP 2) Recluster
    print('Recluster')
    recluster(fl_server,list_clients,row_exp)

    #STEP 3) Merge
    print('Merge')
    fl_server.num_clusters = merge(fl_server,list_clients,lambda_threshold,connection_size_t)

def recluster(fl_server: Server, list_clients : list, row_exp : dict)-> None : 
  """
    Reassign clients to clusters based on their distance to cluster models.

    Arguments:
        fl_server (Server): The federated learning server containing cluster models.
        list_clients (list): A list of client objects involved in federated learning.
        row_exp (dict): A dictionary containing experiment parameters such as the number of clusters.

    Returns:
        None: This function modifies the client cluster assignments in place.
    """
  
  for client in list_clients:
    client_dist_to_clusters = []
    for cluster_id in range(row_exp['num_clusters']):
      client_dist_to_clusters.append(cross_cluster_distance_client_to_cluster(fl_server,list_clients,cluster_id,client))
    client.cluster_id = np.argmin(client_dist_to_clusters)  

def merge(fl_server: Server,list_clients : list, 
          lambda_threshold: float, connection_size_t: int) -> int :
  """
  Creates a graph from a similarity matrix with edges based on a threshold.

  Args:
    list_clients : list of clients present in the federated learning setup
    similarity_matrix: A numpy array representing the similarity matrix of clients models
    lambda_threshold: The threshold for edge inclusion.
    connection_size_t: The minimum size of connected components to include.
  Returns: 
    Number of initial cluster
    """
  
  num_clusters = fl_server.num_clusters
  similarity_matrix = np.zeros((num_clusters, num_clusters))

  G = nx.Graph()
  # Add nodes to the graph
  for cluster_id in range(num_clusters):
    G.add_node(cluster_id)  
    for cluster_id2 in range(cluster_id + 1, num_clusters):  # Avoid double counting
        # Compute the similarity (or distance) between cluster_id and cluster_id2
        similarity = cross_cluster_distance_cluster_to_cluster(fl_server, list_clients, cluster_id, cluster_id2)
        # Fill in the symmetric matrix (i, j) and (j, i)
        similarity_matrix[cluster_id, cluster_id2] = similarity
        similarity_matrix[cluster_id2, cluster_id] = similarity
# Add edges based on the threshold
  for i in range(num_clusters):
    for j in range(i + 1, num_clusters):  # Avoid duplicate edges
      if similarity_matrix[i, j] <= lambda_threshold:
        G.add_edge(i, j)

  # Find connected components
  clusters_list = [c for c in nx.connected_components(G) if len(c) >= connection_size_t]
  if len(clusters_list) > 0 :
    # Set the client cluster id to its corresponding cluster
    for cluster_id in range(len(clusters_list)):
      for client_id in clusters_list[cluster_id]:
          list_clients[client_id].cluster_id = cluster_id
    num_clusters = len(clusters_list)

  return num_clusters

def one_shot(fl_server : Server,list_clients : list, lambda_threshold: float, connection_size_t: int,similarity_matrix : np.ndarray) -> int :
  """
  Creates a graph from a similarity matrix with edges based on a threshold.

  Args:
    list_clients : list of clients present in the federated learning setup
    similarity_matrix: A numpy array representing the similarity matrix of clients models
    lambda_threshold: The threshold for edge inclusion.
    connection_size_t: The minimum size of connected components to include.
  Returns: 
    Number of initial cluster
    """


  print('Doing One Shot Clustering Initialization')
  num_clients = len(list_clients)
  G = nx.Graph()

  # Add nodes to the graph
  for i in range(num_clients):
    G.add_node(i)

  # Add edges based on the threshold
  for i in range(num_clients):
    for j in range(i + 1, num_clients):  # Avoid duplicate edges
      if similarity_matrix[i, j] <= lambda_threshold:
        G.add_edge(i, j)

  # Find connected components
  clusters_list = [c for c in nx.connected_components(G) if len(c) >= connection_size_t]
      
  # Set the client cluster id to its corresponding cluster
  for cluster_id in range(len(clusters_list)):
    for client_id in clusters_list[cluster_id]:
        list_clients[client_id].cluster_id = cluster_id
  return len(clusters_list)

def trimmed_mean_beta_aggregation(model, list_clients, row_exp, beta) -> dict:
    """
    Compute the average parameter updates across multiple clients after trimming outliers 
    using the Trimmed Mean Beta (TMB) method, and aggregate using dataset size as weights.
    
    Arguments:
        model : Server model to update
        list_clients : List of client instances with parameter updates
        beta : Fraction of extreme values to trim (e.g., 0.1 for 10%)
        row_exp : Experiment's global parameters
    Returns:
        model : Updated model to return
    """
    from src.utils_training import train_model
    from src.utils_fed import model_avg
    import torch
    import copy 
    
    if beta < 0 or beta >= 0.5:
        raise ValueError("beta must be in the range [0, 0.5).")

    # Step 1: Train each client and collect their model weights
    for client in list_clients:
        print("Training client ", client.id)
        client.model, _, acc, client.update = train_model(
            client.model, client.data_loader['train'], client.data_loader['val'], row_exp
        )
        client.round_acc.append(acc)

    # Step 2: Create a list of model weights for comparison
    list_client_weights = []

    for client in list_clients:
        
        client_wts = {}

        for name, param in client.model.named_parameters():
                if param.requires_grad:
                    client_wts[name] = param.data.clone()
        list_client_weights.append(client_wts)

    wts_dict_of_lists = {
            key: torch.stack([wt[key] for wt in list_client_weights], dim=0)
            for key in list_client_weights[0].keys()
        }
    
    start_idx = int(beta * len(list_clients))
    end_idx = int((1 - beta) * len(list_clients))
    
    if end_idx <= start_idx + 1 or beta == 0:
        start_idx = 0
        end_idx = len(list_clients)
        avg_wt = {key: item.mean(dim=0) for key, item in wts_dict_of_lists.items()}
    else:
        sorted_dict = {
            key: torch.sort(item, dim=0)[0]
            for key, item in wts_dict_of_lists.items()
        }
        avg_wt = {
            key: item[start_idx:end_idx, ...].mean(dim=0)
            for key, item in sorted_dict.items()
        }
        model = copy.deepcopy(list_clients[0].model)
        #load new averaged model 
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = avg_wt[name] 
    
    return model            


