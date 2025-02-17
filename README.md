Framework Parameters

This document describes the parameters used in the federated learning framework. Each parameter is explained below, along with its possible values and purpose.
Parameters
exp_type, params

    Description: Specifies the type of experiment to run. if wrong value for params use default. 

    Possible Values:

        federated, None : Benchmark experiment for global federated learning. 
        
        fedprox, mu : Federated learning with FedProx optimization. Mu parameter for FedProx default is 0.01

        oracle-centralized, None : Benchmark experiment for personalized centralized learning.
        
        oracle-cfl,None : Personalized federated learning. 

        cfl, None : CFL. Use Kmeans on models weight for clustering
        
        ifca, None: Iterative Federated Clustering Algorithm.

        hcfl, <metric>: Server-side clustered federated learning using agglomerative clustering with a specified model <metric> (possible Values : euclidean, cosine, MADC, EDC). Default use cosine.

        fedgroup, <metric>: FedGroup Algorithm 

        srfca,"(<lambda>,<beta>)": Experiment using the SRFCA algorithm. Successive Refinement Federated Learning Algorihtm. <lambda> is the distance threshold for clustering, <beta> is the trimmed mean parameter
        
        cornflqs, <metric> : cornflqs clustered federated learning. (possible Values : euclidean, cosine, MADC, EDC).

dataset

    Description: The dataset to use for the experiment.

    Possible Values:

        mnist: MNIST dataset.
        
        kmnist: KMNIST dataset.

        fashion-mnist : FASHION-MNIST dataset.

        cifar10: CIFAR-10 dataset.

nn_model

    Description: The neural network model to use for training.

    Possible Values:

        linear: A simple linear model.

        convolutional: A convolutional neural network (CNN) model. Must use for CIFAR-10 dataset.

heterogeneity_type

    Description: The type of data heterogeneity to simulate or test. In a non-IID setup, heterogeneity is defined by subsets of clients where the data distribution is almost IID within each subset but differs between subsets.

        Possible Values:

        features-distribution-skew: Heterogeneity based on differences in feature distributions across clients.

        concept-shift-on-features: Heterogeneity based on changes in the relationship between features and labels (e.g., the same features may correspond to different labels for different clients).

        concept-shift-on-labels: Heterogeneity based on changes in the interpretation of labels across clients (e.g., the same label may represent different concepts for different clients).

skew

    Description: The type of skew applied to the dataset. Skew refers to the uneven distribution of data across clients or heterogeneity classes.

    Possible Values:

        quantity-skew-type-1: Quantity skew within heterogeneity classes (i.e., uneven data distribution among clients within the same heterogeneity class).

        quantity-skew-type-2: Quantity skew between heterogeneity classes (i.e., uneven data distribution across different heterogeneity classes).

        label-skew: Skew based on uneven distribution of labels across clients (e.g., some clients may have more samples of certain labels than others).

num_clients

    Description: The number of clients participating in the federated learning process.

    Possible Values: Any positive integer (e.g., 10, 100).

num_samples_by_label

    Description: The number of samples per label for each client.

    Possible Values: Any positive integer (e.g., 100, 500).

num_clusters

    Description: The number of clusters to form in clustered federated learning.

    Possible Values: Any positive integer (e.g., , 5).

epochs

    Description: The number of epochs to train the model in centralized mode.

    Possible Values: Any positive integer (e.g., 10, 50).

rounds

    Description: The number of communication rounds in federated learning.

    Possible Values: Any positive integer (e.g., 100, 00).

seed

    Description: The random seed for reproducibility.

    Possible Values: Any integer (e.g., 4, 13).
