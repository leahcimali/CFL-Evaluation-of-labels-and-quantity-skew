from pandas import DataFrame
from pathlib import Path
from torch import tensor


def save_histograms(base_path: str) -> None:

    """Read csv files found in the given base_path and generates and saves histogram plots of clients assignments

    Arguments:
        base_path : The base directory path where the csv files are located

    Raises : 
        Warning when the csv file is not of the expected format (code generated results csv)
    """

    import pandas as pd

    pathlist = Path(base_path).rglob('*.csv') 
    
    for file_path in pathlist:

        if 'benchmark' not in str(file_path):
            
            try:

                df_results = pd.read_csv(file_path)

                plot_histogram_clusters(df_results, file_path.stem, base_path)
    
            except Exception as e:
        
                print(f"Error: Unable to open result file {file_path}.", e)
            
                continue
    return



def get_clusters(df_results : DataFrame) -> list:
    
    """ Function to returns a list of clusters ranging from 0 to max_cluster (uses: append_empty_clusters())
    """

    list_clusters = list(df_results['cluster_id'].unique())

    list_clusters = append_empty_clusters(list_clusters)

    return list_clusters


def append_empty_clusters(list_clusters : list) -> list:
    """
    Utility function for ``get_clusters'' to handle the situation where some clusters are empty by appending the clusters ID
    
    Arguments:
        list_clusters: List of clusters with clients

    Returns:
        List of clusters with or without clients
    """

    list_clusters_int = [int(x) for x in list_clusters]
    
    max_clusters = max(list_clusters_int)
    
    for i in range(max_clusters + 1):
        
        if i not in list_clusters_int:
            
            list_clusters.append(str(i))

    return list_clusters



def get_z_nclients(df_results : dict, x_het : list, y_clust : list, labels_heterogeneity : list) -> list:
    
    """ Returns the number of clients associated with a given heterogeneity class for each cluster"""

    z_nclients = [0]* len(x_het)

    for i in range(len(z_nclients)):
        
        z_nclients[i] = len(df_results[(df_results['cluster_id'] == y_clust[i]) &
                                       (df_results['heterogeneity_class'] == labels_heterogeneity[x_het[i]])])

    return z_nclients



def plot_img(img : tensor) -> None:

    """Utility function to plot an image of any shape"""

    from torchvision import transforms
    import matplotlib.pyplot as plt

    plt.imshow(transforms.ToPILImage()(img))



def plot_histogram_clusters(df_results: DataFrame, title: str, base_path: str) -> None:
    """ Function to create 3D Histograms of clients to cluster assignments showing client's heterogeneity class
    with skew values represented as hue in a stacked bar format.
    
    Arguments:
        df_results : DataFrame containing all parameters from the resulting csv files
        title : The plot title. The image is saved in base_path/plots/histogram_' + title + '.png'
        base_path : The base directory path where the plots will be saved
    """
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    # Unique values for heterogeneity_class and skew
    labels_heterogeneity = list(df_results['heterogeneity_class'].unique())
    skew_values = list(df_results['skew'].unique())
    
    bar_width = bar_depth = 0.3

    n_clusters = len(get_clusters(df_results))
    n_heterogeneities = len(labels_heterogeneity)

    # Create a color map based on skew values
    cmap = plt.get_cmap('tab10')  # Choosing a color map
    color_map = {skew: cmap(i / len(skew_values)) for i, skew in enumerate(skew_values)}

    # Set up the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over heterogeneity classes and clusters to create bars
    for h_idx, heterogeneity_class in enumerate(labels_heterogeneity):
        for cluster_id in get_clusters(df_results):
            
            # Filter data for the specific heterogeneity_class and cluster_id
            filtered_data = df_results[
                (df_results['heterogeneity_class'] == heterogeneity_class) & 
                (df_results['cluster_id'] == cluster_id)
            ]
            
            # If there's no data for this combination, skip
            if filtered_data.empty:
                continue

            # Initialize the bottom of the stack
            z_bottom = 0
            
            # Iterate over each skew value
            for skew in skew_values:
                # Count the number of clients for this skew value
                count = len(filtered_data[filtered_data['skew'] == skew])
                
                if count > 0:
                    # Coordinates for the current segment
                    x = h_idx
                    y = int(cluster_id)
                    dx = bar_width
                    dy = bar_depth
                    dz = count  # Height of the current segment
                    
                    # Draw the segment
                    ax.bar3d(x, y, z_bottom, dx, dy, dz, color=color_map[skew])
                    ax.view_init(elev=50)  # Change elev and azim for rotation
                    # Update the bottom for the next segment
                    z_bottom += dz

    # Setting up ticks and labels
    list_clusters = [x for x in 'abcdefghijklmnopqrstuvwxyz'][:n_clusters]

    ticksy = np.arange(0.25, len(list_clusters), 1)
    ticksx = np.arange(0.25, len(labels_heterogeneity), 1)

    plt.xticks(ticksx, labels_heterogeneity)
    plt.yticks(ticksy, list_clusters)

    plt.ylabel('Cluster ID')
    plt.xlabel('Heterogeneity Class')
    
    ax.set_zlabel('Number of Clients')
    
    plt.title(title, fontdict=None, loc='center', pad=None)
    
    # Create a legend for the color mapping
    legend_patches = [Patch(color=color_map[skew], label=skew) for skew in skew_values]
    plt.legend(handles=legend_patches, title="Skew", loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.savefig(Path(base_path) / 'plots' / f'histogram_{title}.png')
    
    plt.close()

    return




def normalize_results(results_accuracy : float, results_std : float) -> int:

    """Utility function to convert float accuracy and std to percentage """
    
    if results_accuracy < 1:
        
        results_accuracy = results_accuracy * 100
    
        results_std = results_std * 100

    return results_accuracy, results_std


def summarize_results(base_path) -> None:

    """ Creates results summary of all the results files under "results/summarized_results.csv"""
        
    from pathlib import Path
    import pandas as pd
    from numpy import mean, std

    from src.metrics import calc_global_metrics
    import sys



    pathlist = Path(base_path).rglob('*.csv')
    list_results = []

    for path in pathlist:
        
        if 'summarized_results' not in str(path):
            
            df_exp_results = pd.read_csv(path)

            results_accuracy = mean(list(df_exp_results['accuracy'])) 
            results_std = std(list(df_exp_results['accuracy']))

            results_accuracy, results_std = normalize_results(results_accuracy, results_std)

            accuracy =  "{:.2f}".format(results_accuracy) + " \\pm " +   "{:.2f}".format(results_std)

            list_params = path.stem.split('_')      

            dict_exp_results = {
            "exp_type": list_params[0],
            "params": list_params[1],
            "dataset": list_params[2],  
            "nn_model": list_params[3],
            "heterogeneity_class": list_params[4],
            "skew": list_params[5],
            "number_of_clients": list_params[6],
            "samples by_client": list_params[7],
            "num_clusters": list_params[8],
            "epochs": list_params[9],
            "rounds": list_params[10],
            "accuracy": accuracy
            }

            try:
                
                labels_true = list(df_exp_results['heterogeneity_class'])
                labels_pred = list(df_exp_results['cluster_id'])
                
                dict_metrics = calc_global_metrics(labels_true=labels_true, labels_pred=labels_pred)
            
                dict_exp_results.update(dict_metrics)
                
            
            except:

                print(f"Warning: Could not calculate cluster metrics for file {path}")
                
    
            list_results.append(dict_exp_results)
            
    df_results = pd.DataFrame(list_results)
    df_results.sort_values(['heterogeneity_class', 'dataset', 'exp_type', 'nn_model','number_of_clients'], inplace=True)

    try: 
        df_results = df_results[['exp_type', "params", 'nn_model', 'number_of_clients', 'dataset', 'num_clusters', 'heterogeneity_class', 'skew', "accuracy", "ARI", "AMI", "hom", "cmplt", "vm"]]
    except KeyError as e: 
        missing_cols = [col for col in ["ARI", "AMI", "hom", "cmplt", "vm"] if col not in df_results.columns]
        for col in missing_cols:
            df_results[col] = "n/a"
        df_results = df_results[['exp_type', "params", 'nn_model', 'number_of_clients', 'dataset', 'num_clusters', 'heterogeneity_class', 'skew', "accuracy", "ARI", "AMI", "hom", "cmplt", "vm"]]
    df_results.sort_values(['dataset', 'skew','exp_type','accuracy'], inplace=True)

    df_results.to_csv(Path(base_path) / "summarized_results.csv", float_format='%.2f', index=False, na_rep="n/a")

    return


def granular_results(base_path) -> None : 
    """
    Processes experimental results to compute and save accuracy statistics by class and skew type.
    Parameters:
    df_exp_results (pd.DataFrame): DataFrame containing experimental results with columns 'heterogeneity_class', 'skew', and 'accuracy'.
    dict_exp_results (dict): Dictionary containing experiment metadata with keys 'exp_type', 'dataset', 'heterogeneity_class', and 'skew'.
    Returns:
    None: The function saves the processed results to a CSV file based on the skew type.
    """
    import pandas as pd
    from pathlib import Path
    from numpy import mean, std
    pathlist = [path for path in Path(base_path).rglob('*.csv') if 'summarized_results' not in str(path) and 'noqs.csv' not in str(path) and 'qs1.csv' not in str(path) and 'qs2.csv' not in str(path)]
    
    for path in pathlist:
        
        df_exp_results = pd.read_csv(path)

        results_accuracy = mean(list(df_exp_results['accuracy'])) 
        results_std = std(list(df_exp_results['accuracy']))

        results_accuracy, results_std = normalize_results(results_accuracy, results_std)

        accuracy =  "{:.2f}".format(results_accuracy) + " \\pm " +   "{:.2f}".format(results_std)

        list_params = path.stem.split('_')      

        dict_exp_results = {
        "exp_type": list_params[0],
        "params": list_params[1],
        "dataset": list_params[2],  
        "nn_model": list_params[3],
        "heterogeneity_class": list_params[4],
        "skew": list_params[5],
        "number_of_clients": list_params[6],
        "samples by_client": list_params[7],
        "num_clusters": list_params[8],
        "epochs": list_params[9],
        "rounds": list_params[10],
        "accuracy": accuracy
        }

                
        accuracy_by_class = df_exp_results.groupby(['heterogeneity_class', 'skew'])['accuracy'].agg(['mean', 'std']).reset_index()
        # Rename columns for clarity
        accuracy_by_class.columns = ['heterogeneity_class', 'skew', 'mean_accuracy', 'std_accuracy']
        # Map heterogeneity_class to class_1, class_2, etc.
        class_mapping = {value: f'class_{i+1}' for i, value in enumerate(accuracy_by_class['heterogeneity_class'].unique())}
        accuracy_by_class['heterogeneity_class'] = accuracy_by_class['heterogeneity_class'].map(class_mapping)
        # Concatenate the heterogeneity_class and skew columns
        accuracy_by_class['class_skew'] = accuracy_by_class['heterogeneity_class'].astype(str) + '_' + accuracy_by_class['skew']
        accuracy_by_class['accuracy'] = accuracy_by_class['mean_accuracy'].round(2).astype(str) + ' \pm ' + accuracy_by_class['std_accuracy'].round(2).astype(str)
        accuracy_by_class.drop(columns=['heterogeneity_class', 'skew','mean_accuracy','std_accuracy'], inplace=True)
        df = accuracy_by_class.pivot_table(index=None, columns='class_skew', values='accuracy', aggfunc='first')
        df.columns.name = None
        df.reset_index(drop=True, inplace=True)
        # Add the exp_type, dataset, heterogeneity_class, and skew to the dataframe
        df['exp_type'] = dict_exp_results['exp_type']
        df['dataset'] = dict_exp_results['dataset']
        df['heterogeneity_class'] = dict_exp_results['heterogeneity_class']
        df['skew'] = dict_exp_results['skew']
        
        # Reorder columns to place the new columns at the beginning
        cols = ['exp_type', 'dataset', 'heterogeneity_class', 'skew'] + [col for col in df.columns if col not in ['exp_type', 'dataset', 'heterogeneity_class', 'skew']]
        df = df[cols]
        granular_results_path = Path('granular_results')
        granular_results_path.mkdir(parents=True, exist_ok=True)
        if dict_exp_results['skew'] == 'quantity-skew-type-1':
            file_path = "granular_results/qs1.csv"
        elif dict_exp_results['skew'] == 'quantity-skew-type-2':
            file_path = "granular_results/qs2.csv"
        else:
            file_path = "granular_results/noqs.csv"

        if Path(file_path).exists():
            df.to_csv(file_path, mode='a', header=False, float_format='%.2f', index=False, na_rep="n/a")
        else:
            df.to_csv(file_path, float_format='%.2f', index=False, na_rep="n/a")
    return



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "results/"
    save_histograms(base_path)
    summarize_results(base_path)
    granular_results(base_path)