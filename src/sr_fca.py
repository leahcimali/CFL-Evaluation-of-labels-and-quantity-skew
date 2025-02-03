import networkx as nx
import numpy as np
from src.fedclass import Server, Client
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy

def hyper_parameters(row_exp)-> tuple: 
  """
    Determine hyperparameters based on the experiment's dataset and heterogeneity type.

    Arguments:
        row_exp (dict): A dictionary containing experiment configurations, 
                        including dataset type and heterogeneity type.

    Returns:
        tuple: A tuple containing the lambda_threshold, connection_size_t, and beta values.
    """
  
  lambda_threshold,connection_size_t,beta = 0.9, 2, 0
  if row_exp['dataset'] == 'mnist':
    if row_exp['heterogeneity_type'] == 'concept-shift-on-features':
      lambda_threshold,connection_size_t,beta = 0.9, 2, 0
    elif row_exp['heterogeneity_type'] == 'features-distribution-skew':
      lambda_threshold,connection_size_t,beta = 1.38, 4, 0
  
  return lambda_threshold,connection_size_t,beta

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
  
  from src.utils_training import test_model
  #SRFCA hyperparameters
  lambda_threshold,connection_size_t,beta = hyper_parameters(row_exp)
  print('Hyper-parameters : ', lambda_threshold,connection_size_t,beta)
  # ONE_SHOT STEP

  row_exp['num_clusters'] = one_shot(fl_server,list_clients,row_exp,lambda_threshold,connection_size_t)
  fl_server.num_clusters = row_exp['num_clusters']
  fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}  
  print('Initialized Clusters : '+ str(fl_server.num_clusters))

  for round in range(row_exp['federated_rounds']):
    print('Communication Round ' + str(round+1))
    # REFINE STEP
    print('Doing Refine step')
    refine_step = True
    if round >2 :
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
    trimmed_mean = trimmed_mean_beta_aggregation(cluster_clients_list,row_exp,beta)
    fl_server.clusters_models[cluster_id] = update_server_model(fl_server.clusters_models[cluster_id],trimmed_mean)
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

def one_shot(fl_server : Server,list_clients : list, row_exp : dict, lambda_threshold: float, connection_size_t: int) -> int :
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
  from src.utils_fed import send_clusters_models_to_clients
  from src.utils_training import train_central
  send_clusters_models_to_clients(list_clients,fl_server)

  # First Training
  for client in list_clients : 
    client.model, _, acc , client.update = train_central(client.model, client.data_loader['train'], client.data_loader['val'],row_exp)
    client.round_acc.append(acc)
  
  # Distance Matrix for ONE-SHOT Step
  similarity_matrix = compute_distance_matrix(list_clients)
  print(similarity_matrix)
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

import torch
import numpy as np

def cross_cluster_distance_clients(client: Client, client2: Client) -> float:
  """
  Calculate the cross-cluster distance between two clients based on the loss of their models.

  The distance is computed as the average of the client model losses on other client training data.

  Arguments:
      client (Client): The first client.
      client2 (Client): The second client.

  Returns:
      float: The cross-cluster distance between the two clients, calculated as half the sum of their model losses.
  """
  from src.utils_fed import loss_calculation
  return 1/2 * (loss_calculation(client.model, client2.data_loader['train']) +
                loss_calculation(client2.model, client.data_loader['train']))
    
def cross_cluster_distance_client_to_cluster(fl_server, list_clients, cluster_id, client: Client) -> float:
  """
    Calculate the cross-cluster distance between a client and a cluster.

    The distance is computed as the average of the client's model loss on the cluster's clients' training data 
    and the cluster model's loss on the client's training data.

    Arguments:
        fl_server: The federated learning server containing cluster models.
        list_clients (list[Client]): A list of all clients.
        cluster_id (int): The identifier of the target cluster.
        client (Client): The client whose distance to the cluster is being calculated.

    Returns:
        float: The cross-cluster distance between the client and the cluster, 
               computed as half the sum of the two loss components.
    """
  from src.utils_fed import loss_calculation
  cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id] 
  return 1/2 * (np.mean([loss_calculation(client.model, cluster_client.data_loader['train']) for cluster_client in cluster_clients_list]) +
                loss_calculation(fl_server.clusters_models[cluster_id], client.data_loader['train']))

def cross_cluster_distance_cluster_to_cluster(fl_server, list_clients, cluster_id, cluster_id2) -> float:
  """
    Calculate the cross-cluster distance between two clusters.

    The distance is computed as the average of the first cluster model’s loss on the second cluster’s clients' training data 
    and the second cluster model’s loss on the first cluster’s clients' training data.

    Arguments:
        fl_server: The federated learning server containing cluster models.
        list_clients (list[Client]): A list of all clients.
        cluster_id (int): The identifier of the first cluster.
        cluster_id2 (int): The identifier of the second cluster.

    Returns:
        float: The cross-cluster distance between the two clusters, 
               computed as half the sum of the two loss components.
    """
  
  from src.utils_fed import loss_calculation
  cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id] 
  cluster2_clients_list = [client for client in list_clients if client.cluster_id == cluster_id2] 

  return 1/2 * (np.mean([loss_calculation(fl_server.clusters_models[cluster_id], cluster_client.data_loader['train']) for cluster_client in cluster2_clients_list]) +
                np.mean([loss_calculation(fl_server.clusters_models[cluster_id2], cluster_client.data_loader['train']) for cluster_client in cluster_clients_list]))
    
def compute_distance_matrix(list_clients: list) -> np.ndarray:
  """
  Compute the distance matrix for a list of clients based on the cross-cluster distance.

  The distance matrix is symmetric, where each entry (i, j) represents the distance between client i and client j.
  The distance is calculated using the `cross_cluster_distance` function.

  Arguments:
      list_clients (list): A list of Client objects, each with a model and a `train_loader`.

  Returns:
      np.ndarray: A symmetric distance matrix of shape (num_clients, num_clients), where each entry (i, j)
                  contains the cross-cluster distance between client i and client j.
  """
  num_clients = len(list_clients)
  distance_matrix = np.zeros((num_clients, num_clients), dtype=float)

  for i in range(num_clients):
      for j in range(i + 1, num_clients):  # We only need to calculate for the upper triangle (matrix is symmetric)
          dist = cross_cluster_distance_clients(list_clients[i], list_clients[j])
          distance_matrix[i, j] = dist
          distance_matrix[j, i] = dist  # Since the distance matrix is symmetric

  return distance_matrix

import torch
import torch.nn as nn

def trimmed_mean_beta_aggregation(list_clients,row_exp, beta)-> dict:
    """
    Compute the average parameter updates across multiple clients after trimming outliers 
    using the Trimmed Mean Beta (TMB) method.
    
    Arguments:
        list_clients : List of client instances with parameter updates
        beta : Fraction of extreme values to trim (e.g., 0.1 for 10%)
    
    Returns:
        avg_update : Dictionary of averaged parameter updates after trimming
    """
    from src.utils_training import train_central
    # Initialize a dictionary to accumulate the updates for each parameter
    avg_update = {}

    # Iterate over the list of clients
    for client in list_clients:
      client.model, _, acc , client.update = train_central(client.model, client.data_loader['train'], client.data_loader['val'],row_exp)
      client.round_acc.append(acc)
      # Iterate over each parameter update in the client's update
      for name, update_tensor in client.update.items():
          if name in avg_update:
              # Accumulate the updates
              avg_update[name].append(update_tensor)
          else:
              # Initialize the accumulator for this parameter
              avg_update[name] = [update_tensor]

    # Now, apply trimming and calculate the average for each parameter update
    num_clients = len(list_clients)
    trim_size = int(beta * num_clients)

    for name in avg_update:
        # Stack the updates into a tensor
        param_updates = torch.stack(avg_update[name])
        
        # Sort the updates along the batch dimension
        sorted_updates, _ = torch.sort(param_updates, dim=0)

        # Trim the top and bottom 'trim_size' updates
        trimmed_updates = sorted_updates[trim_size:num_clients - trim_size]

        # Compute the mean of the remaining updates
        avg_update[name] = torch.mean(trimmed_updates, dim=0)

    return avg_update

def update_server_model(model, update)-> nn.Module:
    """
    Apply the weight update to the model parameters.

    Arguments:
        model : The model to update (client model or server model)
        update : The dictionary containing parameter updates (client.update)

    Returns:
        The model with updated parameters.
    """
    # Iterate over the update dictionary (client.update)
    with torch.no_grad():  # We don't want to track gradients during this operation
        for name, update_tensor in update.items():
            # Ensure that the update key corresponds to the model's parameter
            if name in model.state_dict():
                # Subtract the update from the model's current parameters
                model.state_dict()[name].sub_(update_tensor)

    return model