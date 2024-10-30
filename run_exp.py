import csv
import subprocess
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run experiments based on CSV configuration.")
parser.add_argument("csv_file", type=str, help="Path to the CSV file with experiment configurations")

# Parse the arguments
args = parser.parse_args()

# Open and read the CSV file
with open(args.csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Skip the header (first row)
    next(reader)

    # Iterate over each row (experiment configuration) in the CSV file
    for row in reader:
        # Assigning CSV values to variables
        exp_type, dataset, nn_model, heterogeneity_type, num_clients, num_samples_by_label, num_clusters, centralized_epochs, federated_rounds, seed = row

        # Building the command to run the driver.py script with the corresponding arguments
        command = [
            "python", "driver.py",
            "--exp_type", exp_type,
            "--dataset", dataset,
            "--nn_model", nn_model,
            "--heterogeneity_type", heterogeneity_type,
            "--num_clients", num_clients,
            "--num_samples_by_label", num_samples_by_label,exp_config
            "--num_clusters", num_clusters,
            "--centralized_epochs", centralized_epochs,
            "--federated_rounds", federated_rounds,
            "--seed", seed]

        # Print the command to check it before running (optional)
        print(f"Running command: {' '.join(command)}")

        # Run the command
        subprocess.run(command)
