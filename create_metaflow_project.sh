#!/bin/bash

# Prompt the user for the project name
read -p "Enter the project name: " project_name

# Create the project directories
mkdir -p "$project_name"/{data/{raw,processed,external},notebooks,flows,scripts,tests}
cd "$project_name" || exit

# Create README and environment files
cat <<EOL > README.md
# $project_name

This project is structured to use Metaflow for orchestrating machine learning workflows. Follow the instructions below to set up the environment and run the project.

## Prerequisites

- **Python 3.8** or higher
- **Conda** (optional, but recommended for environment management)
- **Metaflow** and other dependencies

## Setting Up the Environment

### Using Conda

1. **Create a Conda Environment:**

   Navigate to the project directory and create a Conda environment using the \`environment.yml\` file:

   \`\`\`bash
   conda env create -f environment.yml
   \`\`\`

2. **Activate the Environment:**

   Activate the newly created environment:

   \`\`\`bash
   conda activate $project_name
   \`\`\`

### Using Pip

If you prefer to use \`pip\`, follow these steps:

1. **Install Dependencies:**

   Navigate to the project directory and install dependencies using \`pip\`:

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Running the Flows

1. **Navigate to the Project Directory:**

   Open your terminal and navigate to the home directory of your project:

   \`\`\`bash
   cd path/to/your/$project_name
   \`\`\`

2. **Run the Main Flow:**

   Execute the main flow using Python:

   \`\`\`bash
   python scripts/main.py
   \`\`\`

   To resume from the last successful run, use the \`--resume\` flag:

   \`\`\`bash
   python scripts/main.py --resume
   \`\`\`

## Running Tests

To run the unit tests provided in the \`tests\` directory, use the \`unittest\` module:

\`\`\`bash
python -m unittest discover -s tests
\`\`\`

## Using YAML for Configuration

- **Configuration File:**

  Parameters and settings can be specified in the \`config.yml\` file. This allows for easy modification without changing the code directly.

- **Example Configuration:**

  \`\`\`yaml
  parameters:
    learning_rate: 0.01
    batch_size: 32
    num_epochs: 10
  \`\`\`

EOL

echo "name: $project_name" > environment.yml
echo "dependencies:" >> environment.yml
echo "  - python=3.8" >> environment.yml
echo "  - metaflow" >> environment.yml
echo "  - numpy" >> environment.yml
echo "  - pandas" >> environment.yml
echo "  - scikit-learn" >> environment.yml
echo "  - pyyaml" >> environment.yml

echo "metaflow" > requirements.txt
echo "numpy" >> requirements.txt
echo "pandas" >> requirements.txt
echo "scikit-learn" >> requirements.txt
echo "pyyaml" >> requirements.txt

# Create setup script
cat <<EOL > setup.py
from setuptools import setup, find_packages

setup(
    name='$project_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'metaflow',
        'numpy',
        'pandas',
        'scikit-learn',
        'pyyaml'
    ]
)
EOL

# Create a YAML configuration file
cat <<EOL > config.yml
parameters:
  learning_rate: 0.01
  batch_size: 32
  num_epochs: 10
EOL

# Create a utils.py for common utilities
cat <<EOL > flows/utils.py
from metaflow import Flow, namespace

def get_latest_successful_run(flow_name, tag=None):
    """
    Retrieve the latest successful run of a flow, optionally filtered by a tag.
    """
    namespace(None)  # Reset the namespace to ensure all namespaces are considered

    if tag:
        for run in Flow(flow_name).runs():
            if run.successful and tag in run.tags:
                return run
    else:
        run = Flow(flow_name).latest_successful_run
        if run is not None:
            return run

    raise Exception(f"No successful runs found for flow '{flow_name}' with tag '{tag}'" if tag else f"No successful runs found for flow '{flow_name}'")
EOL

# Create a base template for a ParameterFlow to load YAML configuration
cat <<EOL > flows/parameter_flow.py
from metaflow import FlowSpec, step
import yaml

class ParameterFlow(FlowSpec):

    @step
    def start(self):
        """Load parameters from YAML file"""
        with open('config.yml', 'r') as file:
            self.parameters = yaml.safe_load(file)['parameters']
        self.next(self.end)

    @step
    def end(self):
        print("Parameters loaded:", self.parameters)

if __name__ == '__main__':
    ParameterFlow()
EOL

# Create a base template for Metaflow scripts with error catching
cat <<EOL > flows/data_processing_flow.py
from metaflow import FlowSpec, step, catch
from utils import get_latest_successful_run

class DataProcessingFlow(FlowSpec):

    @catch(var='start_failed')
    @step
    def start(self):
        """Load and preprocess data"""
        params = get_latest_successful_run('ParameterFlow', tag=None).data.parameters
        self.batch_size = params['batch_size']
        print(f"Batch size: {self.batch_size}")
        self.data = "raw data"
        self.next(self.clean_data)

    @step
    def clean_data(self):
        if self.start_failed:
            print("Using default values due to parameter retrieval failure.")
        self.cleaned_data = "processed data"
        self.next(self.end)

    @step
    def end(self):
        print("Data processing complete!")

if __name__ == '__main__':
    DataProcessingFlow()
EOL

cat <<EOL > flows/model_flow.py
from metaflow import FlowSpec, step, catch
from utils import get_latest_successful_run

class ModelFlow(FlowSpec):

    @catch(var='start_failed')
    @step
    def start(self):
        """Define the model architecture"""
        params = get_latest_successful_run('ParameterFlow', tag=None).data.parameters
        self.learning_rate = params['learning_rate']
        print(f"Learning rate: {self.learning_rate}")
        self.model = "model architecture"
        self.next(self.end)

    @step
    def end(self):
        if self.start_failed:
            print("Using default learning rate due to parameter retrieval failure.")
        print("Model definition complete!")

if __name__ == '__main__':
    ModelFlow()
EOL

cat <<EOL > flows/train_flow.py
from metaflow import FlowSpec, step, catch
from utils import get_latest_successful_run

class TrainFlow(FlowSpec):

    @catch(var='start_failed')
    @step
    def start(self):
        """Load data and start training"""
        params = get_latest_successful_run('ParameterFlow', tag=None).data.parameters
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        print(f"Training with learning rate: {self.learning_rate}, epochs: {self.num_epochs}")
        self.data = "processed data"
        self.next(self.train_model)

    @step
    def train_model(self):
        if self.start_failed:
            print("Using default values due to parameter retrieval failure.")
        self.trained_model = "trained model"
        self.next(self.end)

    @step
    def end(self):
        print("Training complete!")

if __name__ == '__main__':
    TrainFlow()
EOL

cat <<EOL > flows/evaluate_flow.py
from metaflow import FlowSpec, step, catch
from utils import get_latest_successful_run

class EvaluateFlow(FlowSpec):

    @catch(var='start_failed')
    @step
    def start(self):
        """Load model and evaluate"""
        params = get_latest_successful_run('ParameterFlow', tag=None).data.parameters
        self.model = "trained model"
        self.test_data = "test data"
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        if self.start_failed:
            print("Proceeding with default model and test data due to parameter retrieval failure.")
        self.results = "evaluation results"
        print(f"Evaluation results: {self.results}")
        self.next(self.end)

    @step
    def end(self):
        print("Evaluation complete!")

if __name__ == '__main__':
    EvaluateFlow()
EOL

# Create the main flow with centralized error handling and resume functionality
cat <<EOL > scripts/main.py
import sys
import os
import argparse
from metaflow import Runner, Flow

# Determine the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Change the working directory to the project root
os.chdir(project_root)

# Add the flows directory to the Python path
sys.path.append(os.path.join(project_root, 'flows'))

def run_flow(flow_script, resume=False, **kwargs):
    """
    Helper function to run or resume a Metaflow flow script with the provided parameters.
    
    :param flow_script: The filename of the Metaflow script to run.
    :param resume: Boolean indicating whether to resume a previous run.
    :param kwargs: Additional parameters to pass to the flow.
    :return: The status of the run ('successful' or 'failed').
    """
    flow_name = flow_script.split('/')[-1].replace('.py', '')

    if resume:
        try:
            # Attempt to resume from the latest successful run
            latest_run = Flow(flow_name).latest_successful_run
            if latest_run:
                print(f"Resuming {flow_name} from run ID {latest_run.id}")
                with Runner(flow_script).resume(origin_run_id=latest_run.id, **kwargs) as running:
                    return handle_run_status(running)
            else:
                print(f"No successful run found for {flow_name}. Starting a new run.")
        except Exception as e:
            print(f"Error resuming {flow_name}: {str(e)}")
    
    # Run the flow normally if resume is not possible or not requested
    with Runner(flow_script).run(**kwargs) as running:
        return handle_run_status(running)

def handle_run_status(running):
    """
    Helper function to handle the run status and print outputs.
    
    :param running: The ExecutingRun object from the Runner API.
    :return: The status of the run ('successful' or 'failed').
    """
    if running.status == 'failed':
        print(f'❌ {running.run} failed:')
        print(f'-- stdout --\n{running.stdout}')
        print(f'-- stderr --\n{running.stderr}')
    elif running.status == 'successful':
        print(f'✅ {running.run} succeeded:')
        print(f'-- stdout --\n{running.stdout}')
    return running.status

def main(resume=False):
    # Step 1: Run ParameterFlow
    status = run_flow('flows/parameter_flow.py', resume=resume)
    if status != 'successful':
        return  # Exit the pipeline if this step fails

    # Step 2: Run DataProcessingFlow
    status = run_flow('flows/data_processing_flow.py', resume=resume)
    if status != 'successful':
        return  # Exit the pipeline if this step fails

    # Step 3: Run ModelFlow
    status = run_flow('flows/model_flow.py', resume=resume)
    if status != 'successful':
        return  # Exit the pipeline if this step fails

    # Step 4: Run TrainFlow
    status = run_flow('flows/train_flow.py', resume=resume)
    if status != 'successful':
        return  # Exit the pipeline if this step fails

    # Step 5: Run EvaluateFlow
    status = run_flow('flows/evaluate_flow.py', resume=resume)
    if status != 'successful':
        return  # Exit the pipeline if this step fails

    print("Pipeline executed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Metaflow pipeline with optional resume capability.')
    parser.add_argument('--resume', action='store_true', help='Resume the pipeline from the last successful run')
    args = parser.parse_args()

    # Pass the resume argument to the main function
    main(resume=args.resume)
EOL

# Create test templates
cat <<EOL > tests/test_data_processing.py
import unittest

class TestDataProcessing(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, 1)
EOL

cat <<EOL > tests/test_model.py
import unittest

class TestModel(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, 1)
EOL

cat <<EOL > tests/test_pipeline.py
import unittest

class TestPipeline(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, 1)
EOL

echo "Project structure for '$project_name' created successfully!"
