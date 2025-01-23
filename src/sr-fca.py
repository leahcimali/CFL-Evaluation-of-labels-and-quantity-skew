import networkx as nx
import numpy as np

def one_shot(list_clients : list, similarity_matrix : np.ndarray, lambda_threshold: float, connection_size_t: int) -> None :
  """
  Creates a graph from a similarity matrix with edges based on a threshold.

  Args:
    list_clients : list of clients present in the federated learning setup
    similarity_matrix: A numpy array representing the similarity matrix of clients models
    lambda_threshold: The threshold for edge inclusion.
    connection_size_t: The minimum size of connected components to include.
    """

  num_clients = len(similarity_matrix)
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
  for cluster_id in len(clusters_list):
    for client_id in clusters_list[cluster_id]:
        list_clients[client_id].cluster_id = cluster_id
    
  return 

def trimmed_mean_beta(gradients : list, beta):
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

def cluster_model_update():
  