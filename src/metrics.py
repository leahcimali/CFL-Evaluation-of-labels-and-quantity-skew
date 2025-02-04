from src.fedclass import Client
import numpy as np
def calc_global_metrics(labels_true: list, labels_pred: list) -> dict:

    """ Calculate global metrics based on model weights

    Arguments:
        labels_true : list
            list of ground truth labels
        labels_pred : list
            list of predicted labels to compare with ground truth
    Returns:
        a dictionary containing the following metrics:
         'ARI', 'AMI', 'hom': homogeneity_score, 'cmpt': completness score, 'vm': v-measure
    """

    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, adjusted_mutual_info_score

    homogeneity_score, completness_score, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)

    ARI_score = adjusted_rand_score = adjusted_rand_score(labels_true, labels_pred)

    AMI_score = adjusted_mutual_info_score(labels_true, labels_pred) 
    
    dict_metrics = {"ARI": ARI_score, "AMI": AMI_score, "hom": homogeneity_score, "cmplt": completness_score, "vm": v_measure}
    
    return dict_metrics

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