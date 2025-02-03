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
  #SRFCA hyperparameters
  lambda_threshold,connection_size_t,beta = 0.77, 2, 0.15
  send_clusters_models_to_clients(list_clients,fl_server)
  for client in list_clients : 
    client.model, _, acc , client.gradient = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
    client.round_acc.append(acc)
  
  distance_matrix = compute_distance_matrix(list_clients)
  print(distance_matrix)
  print('Doing One Shot Clustering Initialization')

  # ONE_SHOT STEP

  row_exp['num_clusters'] = one_shot(list_clients, distance_matrix ,lambda_threshold,connection_size_t)
  fl_server.num_clusters = row_exp['num_clusters']
  fl_server.clusters_models= {cluster_id: copy.deepcopy(fl_server.model) for cluster_id in range(row_exp['num_clusters'])}  
  print('Initialized Clusters : '+ str(fl_server.num_clusters))

  for round in range(row_exp['federated_rounds']):
    print('Communication Round ' + str(round+1))
    # REFINE STEP
    print('Doing Refine step')
    refine(fl_server,list_clients,row_exp,beta)

  for client in list_clients :

        acc = test_model(fl_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

  df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

  return df_results 

def refine(fl_server : Server, list_clients : list, row_exp : dict, beta):
  #STEP 1) Trimmed Mean on each cluster 
  print('trimmed Mean Step')
  for cluster_id in range(row_exp['num_clusters']) :
    cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id] 

    #cluster_client_gradients = torch.stack([client.gradient if isinstance(client.gradient, torch.Tensor) 
    #                                    else torch.tensor(client.gradient) for client in cluster_clients_list], dim=0)

    trimmed_mean = trimmed_mean_beta_aggregation(cluster_clients_list,0.1)
    fl_server.clusters_models[cluster_id] = update_server_model(fl_server.clusters_models[cluster_id],trimmed_mean,0.001)

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

import torch
import torch.nn as nn

def trimmed_mean_beta_aggregation(list_clients, beta):
  """
  Aggregate client gradients using Trimmed Mean Beta (TMB) method.
  
  Arguments:
      list_clients : List of client instances with trained models and gradients
      beta : Fraction of extreme values to trim (e.g., 0.1 for 10%)
  
  Returns:
      avg_grad : Averaged gradients after trimming
  """
  # Collect gradients from all clients
  all_gradients = [client.gradient for client in list_clients]
  
  # Initialize list to hold the trimmed mean gradients
  avg_grad = []
  
  # Iterate over each parameter
  for param_idx in range(len(all_gradients[0])):
      # Collect gradients for the current parameter from all clients
      param_gradients = [grad[param_idx] for grad in all_gradients]
      
      # Stack gradients into a tensor
      param_gradients = torch.stack(param_gradients)
      
      # Sort the gradients along the batch dimension
      sorted_gradients, _ = torch.sort(param_gradients, dim=0)
      
      # Calculate the number of gradients to trim
      num_clients = len(list_clients)
      trim_size = int(beta * num_clients)
      
      # Trim the top and bottom 'trim_size' gradients
      trimmed_gradients = sorted_gradients[trim_size:num_clients - trim_size]
      
      # Compute the mean of the remaining gradients
      mean_gradients = torch.mean(trimmed_gradients, dim=0)
      
      # Append the trimmed mean gradient to the result list
      avg_grad.append(mean_gradients)
  
  return avg_grad

def update_server_model(fl_server_model, avg_grad, learning_rate):
  """
  Update the global model using the averaged gradients.
  
  Arguments:
      fl_server_model : The global model to be updated
      avg_grad : Averaged gradients
      learning_rate : Learning rate for the update
  """
  with torch.no_grad():
      for param, grad in zip(fl_server_model.parameters(), avg_grad):
          param -= learning_rate * grad

  return fl_server_model