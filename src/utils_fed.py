from src.fedclass import Server
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def send_server_model_to_client(list_clients : list, my_server : Server) -> None:
    
    """ Function to copy the Server model to client attributes in a FL protocol

    Arguments:
        list_clients : List of Client objects on which to set the parameter `model'
        my_server : Server object with the model to copy
    """

    import copy

    for client in list_clients:
        setattr(client, 'model', copy.deepcopy(my_server.model))

    return


def send_cluster_models_to_clients(list_clients : list , my_server : Server) -> None:
    """ Function to copy Server modelm to clients based on attribute client.cluster_id

    Arguments: 
        list_clients : List of Clients to update
        my_server : Server from which to fetch models
    """

    import copy

    for client in list_clients:
            if client.cluster_id is None:
                setattr(client, 'model', copy.deepcopy(my_server.model))
            else:
                setattr(client, 'model', copy.deepcopy(my_server.clusters_models[client.cluster_id]))
    return 


def model_avg(list_clients : list, ponderation : bool = True) -> nn.Module:
    
    """  Utility function for the fed_avg function which creates a new model
         with weights set to the weighted average of 
    
    Arguments:
        list_clients : List of Client whose models we want to use to perform the weighted average

    Returns:
        New model with weights equal to the weighted average of those in the input Clients list

    """
    
    import copy
    import torch

    new_model = copy.deepcopy(list_clients[0].model)

    total_data_size = sum(len(client.data_loader['train'].dataset) for client in list_clients)

    for name, param in new_model.named_parameters():

        weighted_avg_param = torch.zeros_like(param)
        
        for client in list_clients:

            data_size = len(client.data_loader['train'].dataset)
            if ponderation :
                weight = data_size / total_data_size            
            else :
                weight = 1/ len(list_clients)
            weighted_avg_param += client.model.state_dict()[name] * weight

        param.data = weighted_avg_param #TODO: make more explicit
        
    return new_model
    
    
def fedavg(my_server : Server, list_clients : list, ponderated : bool = True) -> None:
    """
    Implementation of the (Clustered) federated aggregation algorithm with one model per cluster. 
    The code modifies the cluster models `my_server.cluster_models[i]'

    
    Arguments:
        my_server : Server model which contains the cluster models
        list_clients: List of clients, each containing a PyTorch model and a data loader.
        ponderation : If ponderation is True will pondated weight by quantity of data.
    """
    if my_server.num_clusters == None:

        my_server.model = model_avg(list_clients,ponderated)
    
    else : 
         
         for cluster_id in range(my_server.num_clusters):
                      
            cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id]
            
            if len(cluster_clients_list)>0 :  
          
                my_server.clusters_models[cluster_id] = model_avg(cluster_clients_list,ponderated)
    return


def model_weight_matrix(list_clients : list) -> pd.DataFrame:
   
    """ Create a weight matrix DataFrame using the weights of local federated models for use in the server-side CFL 

    Arguments:

        list_clients: List of Clients with respective models
         
    Returns:
        DataFrame with weights of each model as rows
    """

    import numpy as np
    import pandas as pd
    

    model_dict = {client.id : client.model for client in list_clients}

    shapes = [param.data.cpu().numpy().shape for param in next(iter(model_dict.values())).parameters()]
    weight_matrix_np = np.empty((len(model_dict), sum(np.prod(shape) for shape in shapes)))

    for idx, (_, model) in enumerate(model_dict.items()):

        model_weights = np.concatenate([param.data.cpu().numpy().flatten() for param in model.parameters()])
        weight_matrix_np[idx, :] = model_weights

    weight_matrix = pd.DataFrame(weight_matrix_np, columns=[f'w_{i+1}' for i in range(weight_matrix_np.shape[1])])

    return weight_matrix

def model_dissimilarity(client : list, server_model : list) :
    from sklearn.metrics.pairwise import cosine_similarity
    server_weights = torch.cat([param.detach().flatten() for param in server_model.parameters()]).unsqueeze(0).cpu()
    client_weights = torch.cat([param.detach().flatten() for param in client.model.parameters()]).unsqueeze(0).cpu()
    
    return (1 - cosine_similarity(server_weights.numpy(), client_weights.numpy())) / 2

def client_migration(my_server,client):

    client_server_dissimilarity = [(cluster_id,model_dissimilarity(client, server_model)) for cluster_id, server_model in my_server.clusters_models.items()]
    dissimilarity_values = [dissimilarity for _, dissimilarity in client_server_dissimilarity]

    #Find the index of the minimum dissimilarity
    min_index = np.argmin(dissimilarity_values)

    # Use the index to get the corresponding cluster_id
    min_cluster_id, _ = client_server_dissimilarity[min_index]

    # Update the client with the corresponding model and cluster_id
    client.model = my_server.clusters_models[min_cluster_id]
    client.cluster_id = min_cluster_id
        
def k_means_clustering(list_clients : list, num_clusters : int, seed : int, metric : str ='euclidean') -> None:
    """ Performs a k-mean clustering and sets the cluser_id attribute to clients based on the result
    
    Arguments:
        list_clients : List of Clients on which to perform clustering
        num_clusters : Parameter to set the number of clusters needed
        seed : Random seed to allow reproducibility
    """ 
    from sklearn.cluster import KMeans
    weight_matrix = model_weight_matrix(list_clients)
    if metric == 'EDC': 
        weight_matrix = EDC(weight_matrix, num_clusters, seed)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(weight_matrix)

    weight_matrix['cluster'] = kmeans.labels_
    weight_matrix.index = [client.id for client in list_clients]
    clusters_identities = weight_matrix['cluster']
    
    for client in list_clients : 

        setattr(client, 'cluster_id',clusters_identities[client.id])
    
    return 


def calculate_data_driven_measure(pm : np.ndarray) -> np.ndarray:
    ''' Used in the calculation of MADD. 
    Credit -> Adapted from https://github.com/morningD/FlexCFL
    Arguments:
        pm : proximity matrix, usually cosine similarity matrix of local models weights 
    '''
    # pm.shape=(n_clients, n_dims), dm.shape=(n_clients, n_clients)
    n_clients, n_dims = pm.shape[0], pm.shape[1]
    dm = np.zeros(shape=(n_clients, n_clients))
    

    row_pm_matrix = np.repeat(pm[:,np.newaxis,:], n_clients, axis=1)
 
    col_pm_matrix = np.tile(pm, (n_clients, 1, 1))
    absdiff_pm_matrix = np.abs(col_pm_matrix - row_pm_matrix) # shape=(n_clients, n_clients, n_clients)
    # Calculate the sum of absolute differences
    
    # We should mask these values like sim(1,2), sim(2,1) in d(1,2)
    mask = np.zeros(shape=(n_clients, n_clients))
    np.fill_diagonal(mask, 1) # Mask all diag
    mask = np.repeat(mask[np.newaxis,:,:], n_clients, axis=0)
    for idx in range(mask.shape[-1]):
        #mask[idx,idx,:] = 1 # Mask all row d(1,1), d(2,2)...; Actually d(1,1)=d(2,2)=0
        mask[idx,:,idx] = 1 # Mask all 0->n colum for 0->n diff matrix,
    #difference matrix
    dm = np.sum(np.ma.array(absdiff_pm_matrix, mask=mask), axis=-1) / (n_dims-2.0)

    return dm # shape=(n_clients, n_clients)

def MADC(weight_matrix : pd.DataFrame) -> np.ndarray : 
    """Calculate the MADC (Mean Absolute difference of pairwise cosine similarity)
    
    Arguments:
        weight_matrix : weight matrix DataFrame using the weights of local federated models
    """
    from sklearn.metrics.pairwise import cosine_similarity
    cossim_matrix = cosine_similarity(weight_matrix)
    affinity_matrix = pd.DataFrame(calculate_data_driven_measure(cossim_matrix))
    return affinity_matrix

def EDC(weight_matrix : pd.DataFrame, num_clusters : int, seed : int) ->  np.ndarray : 
    """Calculate the euclidean distance of cosine dissimilarity with SVD decomposition
    
    Arguments:
        weight_matrix : weight matrix DataFrame using the weights of local federated models        
        num_clusters : Parameter to set the number of clusters needed
    """
    from sklearn.metrics.pairwise import cosine_similarity 
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=num_clusters, random_state=seed)
    decomp_updates = svd.fit_transform(weight_matrix.T) # shape=(n_params, n_groups)

        # Record the execution time of EDC calculation
    decomposed_cossim_matrix = cosine_similarity(weight_matrix, decomp_updates.T) # shape=(n_clients, n_clients)
    
    return decomposed_cossim_matrix

def Agglomerative_Clustering(list_clients : list, num_clusters : int, clustering_metric :str, seed : int, linkage_type : str='complete') -> None:
    """ Performs a agglomerative clustering and sets the cluser_id attribute to clients based on the result
    
    Arguments:
        list_clients : List of Clients on which to perform clustering
        num_clusters : Parameter to set the number of clusters needed
        clustering_metric : Specify the used metric, choose from 'euclidean', 'cosine' and MADC 
        linkage_type : Specify the linkage type, choosen from ward, complete, average ans single
        seed : Random seed to allow reproducibility        
    """ 
    from sklearn.cluster import AgglomerativeClustering

    weight_matrix = model_weight_matrix(list_clients)
    if clustering_metric == 'MADC': 
        affinity_matrix = MADC(weight_matrix)
        ac = AgglomerativeClustering(num_clusters, metric='precomputed', linkage=linkage_type)
        weight_matrix = affinity_matrix
    else: 
        ac = AgglomerativeClustering(n_clusters=num_clusters,metric=clustering_metric,linkage=linkage_type)
    
    ac.fit(weight_matrix)
    
    weight_matrix['cluster'] = ac.labels_
    weight_matrix.index = [client.id for client in list_clients]
    clusters_identities = weight_matrix['cluster']
    for client in list_clients : 
        setattr(client, 'cluster_id',clusters_identities[client.id])
    return      

def init_server_cluster(my_server : Server, list_clients : list, row_exp : dict, imgs_params: dict, p_expert_opinion : float = 0) -> None:
    
    """ Function to initialize cluster membership for client-side CFL (sets param cluster id) 
    using a given distribution or completely at random. 
    
    Arguments:
        my_server : Server model containing one model per cluster

        list_clients : List of Clients  whose model we want to initialize

        row_exp : Dictionary containing the different global experiement parameters

        p_expert_opintion : Parameter to avoid completly random assignment if neeed (default to 0)
    """
    
    from src.models import GenericLinearModel, GenericConvModel
    import numpy as np
    import copy

    np.random.seed(row_exp['seed'])

    list_heterogeneities = list(dict.fromkeys([client.heterogeneity_class for client in list_clients]))

    if not p_expert_opinion or p_expert_opinion == 0:

        p_expert_opinion = 1 / row_exp['num_clusters']
        
    p_rest = (1 - p_expert_opinion) / (row_exp['num_clusters'] - 1)

    my_server.num_clusters = row_exp['num_clusters']
    if row_exp['nn_model'] == 'linear':
        my_server.clusters_models = {cluster_id: GenericLinearModel(in_size=imgs_params[0]) for cluster_id in range(row_exp['num_clusters'])}
        
    elif row_exp['nn_model'] == 'convolutional':
        my_server.clusters_models = {cluster_id: GenericConvModel(in_size=imgs_params[0], n_channels=imgs_params[1]) for cluster_id in range(row_exp['num_clusters'])}
    
    for client in list_clients:
    
        probs = [p_rest if x != list_heterogeneities.index(client.heterogeneity_class) % row_exp['num_clusters']
                        else p_expert_opinion for x in range(row_exp['num_clusters'])] 

        client.cluster_id = np.random.choice(range(row_exp['num_clusters']), p = probs)
        
        client.model = copy.deepcopy(my_server.clusters_models[client.cluster_id])
    
    return 


def loss_calculation(model : nn.modules, train_loader : DataLoader) -> float:

    """ Utility function to calculate average_loss across all samples <train_loader>

    Arguments:

        model : the input server model
        
        train_loader : DataLoader with the dataset to use for loss calculation
    """ 
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()  
    
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples

    return average_loss




def set_client_cluster(my_server : Server, list_clients : list, row_exp : dict) -> None:
    """ Function to calculate cluster membership for client-side CFL (sets param cluster id)
    
     Arguments:
        my_server : Server model containing one model per cluster

        list_clients : List of Clients  whose model we want to initialize

        row_exp : Dictionary containing the different global experiement parameters
    """

    import numpy as np
    import copy
    
    for client in list_clients:
        
        cluster_losses = []
        
        for cluster_id in range(row_exp['num_clusters']):
        
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'])
        
            cluster_losses.append(cluster_loss)
        
        index_of_min_loss = np.argmin(cluster_losses)
        
        client.model = copy.deepcopy(my_server.clusters_models[index_of_min_loss])
    
        client.cluster_id = index_of_min_loss
    