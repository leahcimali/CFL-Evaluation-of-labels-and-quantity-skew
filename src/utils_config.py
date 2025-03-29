import pandas as pd
import itertools
import click

def get_base_params():
    return {
        'datasets': ['mnist', 'fashion-mnist', 'kmnist', 'cifar10','pathmnist','octmnist','tissuemnist'],
        'heterogeneity_type': ['concept-shift-on-features', 'concept-shift-on-labels', 'features-distribution-skew'],
        'skews': ['None', 'quantity-skew-type-1', 'quantity-skew-type-2'],
        'exp_params_list': [
            ('fedavg', 'None'),
            ('fedprox', 0.1),
            ('cfl', 'None'),
            ('hcfl', 'euclidean'),
            ('fedgroup', 'edc'),
            ('ifca', 'best'),
            ('srfca', 'None'),
            ('cornflqs', 'euclidean')
            #,('oracle-centralized', 'clustering')
        ],
        'seed_values': [42, 43, 44, 45, 46],
        'nn_model': 'linear',
        'number_of_clients': 100,
        'num_samples_by_label': 50,
        'num_clusters': 4,
        'default_epochs': 10,
        'rounds': 20
    }
@click.command()
@click.option('--dataset', default=None, help='Specify dataset to use')
@click.option('--heterogeneity_type', default=None, help='Specify heterogeneity type to use')
@click.option('--skew', default=None, help='Specify skew to use')
@click.option('--filename', default='exp_configs.csv', help='Specify output file name')
def main(dataset, heterogeneity_type, skew, filename):
    params = get_base_params()
    
    datasets = [dataset] if dataset else params['datasets']
    heterogeneity_types = [heterogeneity_type] if heterogeneity_type else params['heterogeneity_types']
    skews = [skew] if skew else params['skews']
    
    base_data = generate_base_data(datasets, heterogeneity_types, skews)
    combinations = generate_combinations(base_data, params)
    save_combinations_to_csv(combinations, filename)

def generate_base_data(datasets, heterogeneity_types, skews):
    return [[dataset, heterogeneity, skew] for dataset in datasets for heterogeneity in heterogeneity_types for skew in skews]

def generate_combinations(base_data, params):
    all_combinations = []
    for base, (exp_type, exp_param), seed in itertools.product(base_data, params['exp_params_list'], params['seed_values']):
        epochs = 100 if exp_type == 'oracle-centralized' else params['default_epochs']
        nn_model_used = 'convolutional' if base[0] in ['cifar10','pathmnist','octmnist','tissuemnist'] else params['nn_model']
        epochs_used = 5 if base[0]in ['cifar10','pathmnist','octmnist','tissuemnist'] else epochs
        rounds_used = 100 if base[0] in ['cifar10','pathmnist','octmnist','tissuemnist'] else params['rounds']
        num_samples_by_labels_used = params['num_samples_by_label']
        combination = base + [exp_type, exp_param, seed, nn_model_used, 
                      params['number_of_clients'], num_samples_by_labels_used, params['num_clusters'], epochs_used, rounds_used]
        all_combinations.append(combination)
    return all_combinations

def save_combinations_to_csv(combinations, filename):
    columns = ['dataset', 'heterogeneity_type', 'skew', 'exp_type', 'params', 'seed', 'nn_model', 'number_of_clients', 'num_samples_by_label', 'num_clusters', 'epochs', 'rounds']
    df_combinations = pd.DataFrame(combinations, columns=columns)
    df_combinations = df_combinations[['exp_type', 'params', 'dataset', 'nn_model', 'heterogeneity_type', 'skew', 'number_of_clients', 'num_samples_by_label', 'num_clusters', 'epochs', 'rounds', 'seed']]
    df_combinations.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
