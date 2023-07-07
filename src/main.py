import argparse
import yaml
from experiments.autoencoder import train
from experiments.autoencoder import generate
from utils.dataset import load_data
CONFIG_FILE_PATH = 'configs/main.yaml'

# Command line argument parser
parser = argparse.ArgumentParser(description='Run an experiment.')
parser.add_argument('--experiment', type=str, help='The name of the experiment to run.')
parser.add_argument('--mode', type=str, help='The mode to run the experiment.')
parser.add_argument('--filePath', type=str, help='The path to the file to be processed.')

# Parse the command line arguments
args = parser.parse_args()

# Parse the main configuration file
with open(CONFIG_FILE_PATH, 'r') as file:
    config_main = yaml.safe_load(file)

if not config_main['warnings']:
    import warnings
    warnings.filterwarnings("ignore")

# If arguments are not passed, take from config
if args.experiment is None:
    args.experiment = config_main['experiment_name']

if args.mode is None:
    args.mode = config_main['mode']

# Parse the experiment configuration file
if args.filePath is None:
    args.filePath = f'configs/{args.experiment}/{args.mode}.yaml'

try:
    with open(args.filePath, 'r') as file:
        config_experiment = yaml.safe_load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"""
        Configuration file for experiment {args.experiment} and mode {args.mode} not found.
        Please make sure that the file {args.mode}.yaml exists in the folder configs/{args.experiment}.
        Note that it is not possible to pass individual experiment parameters from the command line,
        a configuration file is required.
        """)


# Depending on the mode and the experiment name, execute the appropriate code
# List of supported experiments
supported_experiments = ["autoencoder", "clip", "meshGenerator"]

# Check if the provided experiment is supported
if args.experiment in supported_experiments:
    data = load_data(config_experiment["dataset"], config_experiment)
    if args.mode == "train":
        train(data, config_experiment)
    elif args.mode == "generate":
        generate(data, config_experiment)
    else:
        raise ValueError(f"Mode {args.mode} is not supported for experiment {args.experiment}.")
else:
    raise ValueError(f"Experiment {args.experiment} is not supported.")

    
