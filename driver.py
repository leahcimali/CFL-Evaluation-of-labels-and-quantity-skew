import os
import traceback

# Set the environment variable for deterministic behavior with CuBLAS (Give reproductibility with CUDA) 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import click

@click.command()
@click.option('--exp_type', help="The experiment type to run")
@click.option('--dataset')
@click.option('--nn_model', help= "The training model to use ('linear (default) or 'convolutional')")
@click.option('--heterogeneity_type', help="The data heterogeneity to test (or dataset)")
@click.option('--skew', help="Type of skew applied to experiment")
@click.option('--num_clients', type=int)
@click.option('--num_samples_by_label', type=int)
@click.option('--num_clusters', type=int)
@click.option('--centralized_epochs', type=int)
@click.option('--federated_rounds', type=int)
@click.option('--seed', type=int)


def main_driver(exp_type, dataset, nn_model, heterogeneity_type, skew, num_clients, num_samples_by_label, num_clusters, centralized_epochs, federated_rounds, seed):

    from pathlib import Path
    import pandas as pd

    from src.utils_data import setup_experiment, get_uid 

    row_exp = pd.Series({"exp_type": exp_type, "dataset": dataset, "nn_model" : nn_model, "heterogeneity_type": heterogeneity_type, "skew": skew, "num_clients": num_clients,
               "num_samples_by_label": num_samples_by_label, "num_clusters": num_clusters, "centralized_epochs": centralized_epochs,
               "federated_rounds": federated_rounds, "seed": seed})
    

    output_name =  row_exp.to_string(header=False, index=False, name=False).replace(' ', "").replace('\n','_')
    
    hash_outputname = get_uid(output_name)

    pathlist = Path("results").rglob('*.json')

    for file_name in pathlist:

        if get_uid(str(file_name.stem)) == hash_outputname:

            print(f"Experiment {str(file_name.stem)} already executed in with results in \n {output_name}.json")   
        
            return 
    try:
        
        model_server, list_clients = setup_experiment(row_exp)
    
    except Exception as e:

        print(f"Could not run experiment with parameters {output_name}. Exception {e}")
        print("Exception details:")
        traceback.print_exc()  # This will print the full traceback

        return 
    
    launch_experiment(model_server, list_clients, row_exp, output_name)

    return          


def launch_experiment(model_server, list_clients, row_exp, output_name, save_results = True):
        
        from src.utils_training import run_cfl_client_side, run_cfl_server_side, run_cfl_hybrid
        from src.utils_training import run_benchmark

        str_row_exp = ':'.join(row_exp.to_string().replace('\n', '/').split())

        if row_exp['exp_type'] == "global-federated" or row_exp['exp_type'] == "pers-centralized" or row_exp['exp_type'] == "fedprox" or row_exp['exp_type'].split('-')[0] == "fedprox":

            print(f"Launching benchmark experiment with parameters:\n{str_row_exp}")   

            df_results = run_benchmark(model_server, list_clients, row_exp)
        
        elif row_exp['exp_type'] == "pers-federated":
            df_results = run_cfl_server_side(model_server, list_clients, row_exp,algorithm='cheat',clustering_metric='none')
        elif row_exp['exp_type'] == "hybrid":
            print(f"Launching hybrid CFL experiment with parameters:\n {str_row_exp}")
            # Need to add other than KMeans
            df_results = run_cfl_hybrid(model_server,list_clients,row_exp)
        elif row_exp['exp_type'] == "client":
            
            print(f"Launching client-side experiment with parameters:\n {str_row_exp}")

            df_results = run_cfl_client_side(model_server, list_clients, row_exp)
        elif row_exp['exp_type'] == "server-nonponderated":
            df_results = run_cfl_server_side(model_server, list_clients, row_exp,ponderated=False)
            
        elif row_exp['exp_type'].split('-')[0] == "server":

            print(f"Launching server-side experiment with parameters:\n {str_row_exp}")
            # exp_type values should be server for Kmeans or 
            # server-agglomerative-euclidean, server-agglomerative-cosine,  server-agglomerative-MADC  
            if len(row_exp['exp_type'].split('-')) == 1 :
                print('Using Kmeans Clustering!')
                df_results = run_cfl_server_side(model_server, list_clients, row_exp)
            else : 
                print('Using Agglomerative Clustering!')
                algorithm = row_exp['exp_type'].split('-')[1]
                clustering_metric = row_exp['exp_type'].split('-')[2]
                df_results = run_cfl_server_side(model_server, list_clients, row_exp,algorithm,clustering_metric)
        elif row_exp['exp_type'].split('-')[0] == "iterative":
            #iterative server-side
            print(f"Launching server-side experiment with parameters:\n {str_row_exp}")
            # exp_type values should be server for Kmeans or 
            # server-agglomerative-euclidean, server-agglomerative-cosine,  server-agglomerative-MADC  
            if len(row_exp['exp_type'].split('-')) == 2 :
                print('Using Kmeans Clustering!')
                df_results = run_cfl_server_side(model_server, list_clients, row_exp,iterative=True)
            elif row_exp['exp_type'].split('-')[2] == 'EDC'  :
                print('Using Kmeans Clustering!')
                df_results = run_cfl_server_side(model_server, list_clients, row_exp,clustering_metric = 'EDC',iterative=True)
            else : 
                print('Using Agglomerative Clustering!')
                algorithm = row_exp['exp_type'].split('-')[2]
                clustering_metric = row_exp['exp_type'].split('-')[3]
                df_results = run_cfl_server_side(model_server, list_clients, row_exp,algorithm,clustering_metric,iterative=True)         
        else:
            
            str_exp_type = row_exp['exp_type']
            
            raise Exception(f"Unrecognized experiement type {str_exp_type}. Please check config file and try again.")
        
        if save_results : 

            df_results.to_csv("results/" + output_name + ".csv")

        return


if __name__ == "__main__":
    main_driver()
