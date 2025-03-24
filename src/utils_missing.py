import os
import pandas as pd
import click

@click.command()
@click.argument('csv_file')
def main(csv_file):
    """
    Reads the CSV file and generates expected filenames by replacing commas with underscores.
    """
    existing_filenames = set(os.listdir('results/'))

    with open(csv_file, 'r') as file:
        with open('missing_experiments.csv', 'w') as f:
            for line in file:
                check_line = "_".join(line.strip().split(',')) + ".csv"
                if check_line not in existing_filenames:
                    f.write(line)
                    
if __name__ == "__main__":
    main()