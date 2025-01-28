import networkx as nx
import numpy as np
from src.fedclass import Server, Client
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy

def srfca(fl_server : Server, list_clients : list, row_exp : dict) -> pd.DataFrame:
  from src.utils_fed import send_clusters_models_to_clients, model_weight_matrix
  from src.utils_training import train_central,test_model
  from src.sr_fca import one_shot
  lambda_threshold,connection_size_t,beta = 0.7, 2, 0.05
  send_clusters_models_to_clients(list_clients,fl_server)
  for client in list_clients : 
    client.model, _, acc , client.gradient = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
    client.round_acc.append(acc)
  
  distance_matrix = compute_distance_matrix(list_clients)
  print(distance_matrix)
  print('Doing One Shot Clustering Initialization')
  row_exp['num_clusters'] = one_shot(list_clients, distance_matrix ,lambda_threshold,connection_size_t)
  setattr(fl_server, 'num_clusters', row_exp['num_clusters'])
  fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}  
  
  for round in range(row_exp['federated_rounds']):
    print('Refine step')
    refine(fl_server,list_clients,row_exp,beta)

  for client in list_clients :

        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

  df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

  return df_results 

def refine(fl_server : Server, list_clients : list, row_exp : dict, beta):
  #STEP 1) Trimmed Mean on each cluster 
  print('trimmed Mean')
  for cluster_id in range(row_exp['num_clusters']) :
    cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id] 
    cluster_client_gradients = [client.gradient for client in cluster_clients_list]
    optimizer = torch.optim.SGD(fl_server.clusters_models[cluster_id].parameters(), lr =0.001)
    print(len(cluster_client_gradients))
    print(type(cluster_client_gradients[0]))
    fl_server.clusters_models[cluster_id] = update_model_with_trimmed_gradients(fl_server.clusters_models[cluster_id],cluster_client_gradients,beta,optimizer)
  #STEP 2) Recluster
  print('Recluster')
  recluster(fl_server,list_clients,row_exp)
  #STEP 3) Merge clusters
  # TO DO 
  
def recluster(fl_server, list_clients : list, row_exp : dict) : 
  for client in list_clients:
    client_dist_to_clusters = []
    print('num cluster   ', row_exp['num_clusters'])
    for cluster_id in range(row_exp['num_clusters']):
      print(client_dist_to_clusters)
      client_dist_to_clusters.append(cross_cluster_distance_client_to_cluster(fl_server,list_clients,cluster_id,client))
    client.cluster_id = np.argmin(client_dist_to_clusters)  

def merge(fl_server, list_clients : list, row_exp : dict, lambda_threshold,connection_size_t) : 
  # To Do
  return
def one_shot(list_clients : list, similarity_matrix : np.ndarray, lambda_threshold: float, connection_size_t: int) -> int :
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
  print('cluster list len ', len(clusters_list))
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
    from src.utils_fed import loss_calculation
    cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id] 

    return 1/2 * (np.mean([loss_calculation(client.model, cluster_client.data_loader['train']) for cluster_client in cluster_clients_list]) +
                  loss_calculation(fl_server.clusters_models[cluster_id], client.data_loader['train']))

def cross_cluster_distance_cluster_to_cluster(fl_server, list_clients, cluster_id, cluster_id2) -> float:
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
  
'''
def trimmed_mean(gradients : list, beta):
  """
  Calculates the trimmed mean of a list of gradients.

  Args:
    gradients: A list of gradient vectors (numpy arrays).
    beta: The trimming fraction (0 <= beta < 1/2).

  Returns:
    The trimmed mean of the gradients.
  """

  num_gradients = len(gradients)
  if num_gradients == 0:
    raise ValueError("Gradients list is empty.")

  # Calculate the number of gradients to trim from each end
  num_trim = int(beta * num_gradients / 2)

  # Sort gradients based on their magnitude (L2 norm)
  sorted_gradients = sorted(gradients, key=lambda x: np.linalg.norm(x))

  # Trim the gradients
  trimmed_gradients = sorted_gradients[num_trim:-num_trim]

  # Calculate the trimmed mean
  trimmed_grad_mean = np.mean(trimmed_gradients, axis=0)

  return trimmed_grad_mean  
'''

def grad_trimmed_mean(gradients: list, beta : float):
    """
    Calculates the trimmed mean of a list of gradients.

    Args:
        gradients: A list of gradient tensors (PyTorch tensors).
        beta: The trimming fraction (0 <= beta < 1/2).

    Returns:
        The trimmed mean of the gradients as a PyTorch tensor.
    """
    num_gradients = len(gradients)
    if num_gradients == 0:
        raise ValueError("Gradients list is empty.")

    # Calculate the number of gradients to trim from each end
    num_trim = int(beta * num_gradients / 2)

    # Sort gradients based on their magnitude (L2 norm)
    sorted_gradients = sorted(gradients, key=lambda x: torch.norm(x))

    # Trim the gradients
    trimmed_gradients = sorted_gradients[num_trim:-num_trim]

    # Calculate the trimmed mean
    trimmed_grad_mean = torch.mean(torch.stack(trimmed_gradients), dim=0)

    return trimmed_grad_mean
  
def update_model_with_trimmed_gradients(model: nn.Module, gradients_list: list, beta, optimizer: optim.Optimizer):
    """
    Updates the model parameters using the trimmed mean of the gradients.

    Args:
        model: The PyTorch model to update.
        gradients_list: A list of gradients for each parameter of the model.
        beta: The trimming fraction (0 <= beta < 1/2).
        optimizer: The optimizer used to update the model parameters. Usually SGD. 
    """
    # Iterate over each parameter in the model
    for param, gradients in zip(model.parameters(), zip(*gradients_list)):
        # Compute the trimmed mean of the gradients for this parameter
        trimmed_grad = grad_trimmed_mean(gradients, beta)

        # Update the parameter's gradient
        param.grad = trimmed_grad

    # Perform a step of the optimizer to update the model parameters
    optimizer.step()
    
    return model
  