import os
import pandas as pd
import click

def get_expected_filenames(csv_file):
    """
    Reads the CSV file and generates expected filenames by replacing commas with underscores.
    """
    with open('config.csv', 'r') as file:
        lines = file.readlines()
        expected_filenames = ["_".join(line.strip().split(',')) + ".csv" for line in lines[1:]]  # Skip the header
    return set(expected_filenames)

@click.command()
@click.argument('csv_file')
def main(csv_file):
    """
    Checks which expected experiment files are missing in the tracking folder.
    """
    tracking_folder = 'results/'
    output_file = os.path.join('./', 'missing_experiments.csv')
    
    expected_filenames = get_expected_filenames(csv_file)
    existing_filenames = set(os.listdir(tracking_folder))
    missing_filenames = expected_filenames - existing_filenames
    missing_filenames = {filename.replace('_', ',').replace('.csv', '') for filename in missing_filenames}
    with open('missing_experiments.csv', 'w') as f:
        # Write the header
        f.write("exp_type,dataset,nn_model,heterogeneity_type,skew,num_clients,num_samples_by_label,num_clusters,epochs,rounds,seed\n")
        for filename in missing_filenames:
            f.write(f"{filename}\n")
    
    print(f"Missing experiments saved to {output_file}")

if __name__ == "__main__":
    main()