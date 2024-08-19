# Project Setup and Execution

This project is structured to use Metaflow for orchestrating machine learning workflows. Follow the instructions below to set up the environment, build the Metaflow UI, and run the project.

## Prerequisites

Before you begin, ensure that the following are installed on your system:

- **Python 3.8** or higher
- **Conda** (recommended for environment management)
- **Docker** and **Docker Compose**

## 1. Set Up the Conda Environment

1. **Create a Conda Environment**:

   Open your terminal and create a new Conda environment with any name you prefer (replace `<env_name>` with your desired environment name):

   ```bash
   conda create -n <env_name> python=3.8
   ```

2. **Activate the Conda Environment:**:
   Activate the newly created environment:

   ```bash
   conda activate <env_name>
   ```
   
## 2. Clone the Metaflow Server Repository
   Clone the Metaflow server repository to your local machine:

   ```bash
   git clone https://github.com/eldorabdukhamidov/metaflow-server.git
   cd metaflow-server
   ```

## 3. Build the Metaflow UI
   To build and set up the Metaflow UI, execute the provided bash script:

   ```bash
   bash install_metaflow-ui.sh
   ```
  This script will set up the necessary Docker containers and services to run the Metaflow UI.

## 4. Verify the Metaflow UI Setup
   After building the UI, you can verify that everything is working by running the provided sample script. Use the following command to run the _sample_pipe.py_ example:
   ```bash
    METAFLOW_SERVICE_URL=http://localhost:8080 \
    METAFLOW_DEFAULT_METADATA="service" \
    METAFLOW_DEFAULT_DATASTORE=s3 \
    METAFLOW_DATASTORE_SYSROOT_S3=s3://metaflow \
    METAFLOW_S3_ENDPOINT_URL=http://localhost:9000/ \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    python sample_pipe.py run
   ```
  Once the script runs successfully, open your web browser and navigate to http://localhost:8083 to view the Metaflow UI.

## 5. Create a New Project
  To create a new Metaflow project with a template, run the following script:
  ```bash
  bash create_metaflow_project.sh
  ```
  This script will generate a project template with Metaflow, including directories for data, flows, scripts, and tests, as well as a configuration file and example flows.

## 6. Run the Project
   After creating the project, you can run it using the following command:
   ```bash
   METAFLOW_SERVICE_URL=http://localhost:8080 \
  METAFLOW_DEFAULT_METADATA="service" \
  METAFLOW_DEFAULT_DATASTORE=s3 \
  METAFLOW_DATASTORE_SYSROOT_S3=s3://metaflow \
  METAFLOW_S3_ENDPOINT_URL=http://localhost:9000/ \
  AWS_ACCESS_KEY_ID=minioadmin \
  AWS_SECRET_ACCESS_KEY=minioadmin \
  python scripts/main.py
   ```
  This command will execute the main flow of the project, which has been generated by the _**create_metaflow_project.sh**_ script.

## 7. Check the Metaflow UI
   As the project runs, you can monitor the progress and view the results in the Metaflow UI by navigating to http://localhost:8083.
