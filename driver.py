import os
import traceback

from logging import getLogger
from src.utils_logging import setup_logging, cprint

# Set the environment variable for deterministic behavior with CuBLAS (Give reproductibility with CUDA) 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import click

@click.command()
@click.option('--exp_type', help="The experiment type to run")
@click.option('--params', help="Parameters specific to exp type, if in bad format or void, will use defaults")
@click.option('--dataset')
@click.option('--nn_model', help= "The training model to use ('linear (default) or 'convolutional')")
@click.option('--heterogeneity_type', help="The data heterogeneity to test (or dataset)")
@click.option('--skew', help="Type of skew applied to experiment")
@click.option('--num_clients', type=int)
@click.option('--num_samples_by_label', type=int)
@click.option('--num_clusters', type=int)
@click.option('--epochs', type=int)
@click.option('--rounds', type=int)
@click.option('--seed', type=int)


def main_driver(exp_type,params, dataset, nn_model, heterogeneity_type, skew, num_clients, num_samples_by_label, num_clusters, epochs, rounds, seed):

    from pathlib import Path
    import pandas as pd

    from src.utils_data import setup_experiment, get_uid, global_test_data
    # To ADD as a parameter in config file in the future
    import os

    # List of folders to ensure exist
    folders = ["granular_results", "results", "datasets", "tracking"]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    
    # set a global test set for all clients in the same heterogeneity type
    # TO DO for local.  
    test = 'global'

    row_exp = pd.Series({"exp_type": exp_type, "params": params, "dataset": dataset, "nn_model" : nn_model, "heterogeneity_type": heterogeneity_type, "skew": skew, "num_clients": num_clients,
               "num_samples_by_label": num_samples_by_label, "num_clusters": num_clusters, "epochs": epochs,
               "rounds": rounds, "seed": seed})
    

    output_name =  row_exp.to_string(header=False, index=False, name=False).replace(' ', "").replace('\n','_')
    
    hash_outputname = get_uid(output_name)
    setup_logging(output_name)
    pathlist = Path("results").rglob('*.json')

    for file_name in pathlist:

        if get_uid(str(file_name.stem)) == hash_outputname:

            cprint(f"Experiment {str(file_name.stem)} already executed in with results in \n {output_name}.json")   
        
            return 
    try:
        
        fl_server, list_clients = setup_experiment(row_exp)
        
        # Update the test data to global 
        if test == 'global':
            global_test_data(list_clients, row_exp)
        
    except Exception as e:

        cprint(f"Could not run experiment with parameters {output_name}. Exception {e}")
        cprint("Exception details:")
        traceback.print_exc()  # This will cprint the full traceback

        return 
    if row_exp['params'] == 'multiple':
        for model_seed in range(42,47):
            cprint(f"Launching experiment with model seed {model_seed}")
            row_exp = pd.Series({"exp_type": exp_type, "params": model_seed, "dataset": dataset, "nn_model" : nn_model, "heterogeneity_type": heterogeneity_type, "skew": skew, "num_clients": num_clients,
               "num_samples_by_label": num_samples_by_label, "num_clusters": num_clusters, "epochs": epochs,
               "rounds": rounds, "seed": seed})
                

            output_name =  row_exp.to_string(header=False, index=False, name=False).replace(' ', "").replace('\n','_')
    
            hash_outputname = get_uid(output_name)
            setup_logging(hash_outputname)

            pathlist = Path("results").rglob('*.json')

            launch_experiment(fl_server, list_clients, row_exp, output_name)

    else :
        launch_experiment(fl_server, list_clients, row_exp, output_name)

    return          


def launch_experiment(fl_server, list_clients, row_exp, output_name, save_results = True):
        
        from src.utils_training import run_cfl_IFCA, run_cfl_server_side, run_cfl_cornflqs, run_fedgroup,run_benchmark, run_srfca
        from src.utils_data import setup_experiment
        str_row_exp = ':'.join(row_exp.to_string().replace('\n', '/').split())

        if row_exp['exp_type'] == "fedavg" or row_exp['exp_type'] == "oracle-centralized" or row_exp['exp_type'] == "fedprox":

            cprint(f"Launching benchmark experiment with parameters:\n{str_row_exp}")   

            df_results, df_tracking =run_benchmark(fl_server, list_clients, row_exp)
        
        elif row_exp['exp_type'] =='srfca': 
            df_results, df_tracking =run_srfca(fl_server,list_clients,row_exp)
        
        elif row_exp['exp_type'] == "oracle-cfl":
            df_results, df_tracking =run_cfl_server_side(fl_server, list_clients, row_exp,algorithm='oracle-cfl',clustering_metric='none')
        
        elif row_exp['exp_type'] == "cornflqs":
            cprint(f"Launching cornflqs CFL experiment with parameters:\n {str_row_exp}")
            # Use agglomerative HC + euclidean distance and ward linkage
            if row_exp['params'] == 'test':
                ponderated = True
                row_exp['params'] = 'euclidean'
            else : 
                ponderated = False
            df_results, df_tracking = run_cfl_cornflqs(fl_server,list_clients,row_exp,ponderated=ponderated)
            
            '''
            TO DO if need add more metrics
            if row_exp['params']== 'edc':
                df_results, df_tracking =run_cfl_cornflqs(fl_server,list_clients,row_exp,algorithm = 'kmeans', clustering_metric='edc')
            elif row_exp['params']== 'madc':
                df_results, df_tracking =run_cfl_cornflqs(fl_server,list_clients,row_exp,algorithm = 'agglomerative', clustering_metric='madc')
            '''

        elif row_exp['exp_type'] == "ifca":
            if row_exp['params'] == "best":
                cprint('Lauching IFCA with multiple seeds!')
                best_accuracy = 0
                for seed in range(42,47):
                    row_exp['params'] = seed
                    fl_server, list_clients = setup_experiment(row_exp)
                    str_row_exp = ':'.join(row_exp.to_string().replace('\n', '/').split())
                    cprint(f"Launching client-side experiment with parameters:\n {str_row_exp}")
                    df, df_track = run_cfl_IFCA(fl_server, list_clients, row_exp)
                    if df['validation'].mean() > best_accuracy:
                        best_accuracy = df['validation'].mean()
                        df_results, df_tracking = df, df_track
                    
            elif isinstance(row_exp['params'], int)  :
                fl_server, list_clients = setup_experiment(row_exp)
                df_results, df_tracking =run_cfl_IFCA(fl_server, list_clients, row_exp)

            else : 
                cprint(f"Launching client-side experiment with parameters:\n {str_row_exp}")
                df_results, df_tracking =run_cfl_IFCA(fl_server, list_clients, row_exp)
        
        elif row_exp['exp_type'] == "cfl":

            cprint(f"Launching server-side experiment with parameters:\n {str_row_exp}")
            
            cprint('Using Kmeans Clustering!')
            df_results, df_tracking =run_cfl_server_side(fl_server, list_clients, row_exp)
        
        elif row_exp['exp_type'] == 'hcfl': 
            cprint('Using Agglomerative Clustering!')
            
            df_results, df_tracking =run_cfl_server_side(fl_server, list_clients, row_exp)
        
        elif row_exp['exp_type'] == "fedgroup":
            #iterative server-side
            cprint(f"Launching FedGroup experiment with parameters:\n {str_row_exp}")
            
            if row_exp['params'] == 'madc':
                cprint(f"Launching FedGroup experiment with parameters:\n {str_row_exp}")
            
                df_results, df_tracking =run_fedgroup(fl_server, list_clients, row_exp, alpha = row_exp["num_clients"], algorithm= 'agglomerative', clustering_metric = 'madc')
            
            else :
                
                row_exp['params'] = 'edc'
                str_row_exp = ':'.join(row_exp.to_string().replace('\n', '/').split())
                cprint(f"Launching FedGroup experiment with parameters:\n {str_row_exp}")

                df_results, df_tracking =run_fedgroup(fl_server, list_clients, row_exp, alpha = row_exp["num_clients"], algorithm= 'kmeans', clustering_metric = 'edc')
        else:
            
            str_exp_type = row_exp['exp_type']
            
            raise Exception(f"Unrecognized experiement type {str_exp_type}. Please check config file and try again.")
        
        if save_results : 
            # Ensure the results and tracking directories exist
            os.makedirs("results", exist_ok=True)
            os.makedirs("tracking", exist_ok=True)

            df_results.to_csv("results/" + output_name + ".csv")
            df_tracking.to_csv("tracking/" + output_name + "_tracking.csv")

        return

if __name__ == "__main__":
    main_driver()