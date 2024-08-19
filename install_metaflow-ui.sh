#!/bin/bash

# Function to clean up and exit on failure
cleanup() {
    echo "An error occurred. Stopping and removing Docker containers..."
    docker-compose down
    exit 1
}

# Clone the repository if it doesn't exist
if [ -d "metaflow-server" ]; then
    echo "Directory 'metaflow-server' already exists."
else
    git clone https://github.com/eldorabdukhamidov/metaflow-server.git || cleanup
fi

# Navigate to the directory
cd metaflow-server || cleanup

# Create .env file
cat <<EOT > .env
# Metaflow
MF_METADATA_DB_HOST=mf-postgres
MF_METADATA_DB_PORT=5432
MF_MIGRATION_ENDPOINTS_ENABLED=1
MF_METADATA_PORT=8080
MF_METADATA_HOST=0.0.0.0
MF_MIGRATION_PORT=8082
METAFLOW_DEFAULT_DATASTORE=s3
METAFLOW_DATASTORE_SYSROOT_S3=s3://metaflow
METAFLOW_S3_ENDPOINT_URL=http://mf-minio:9000/

# Postgres
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=metaflowdb

# Minio
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
EOT

# Source the .env file
source .env || cleanup

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found, installing..."
    sudo apt-get update || cleanup
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common || cleanup
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - || cleanup
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" || cleanup
    sudo apt-get update || cleanup
    sudo apt-get install -y docker-ce docker-compose || cleanup
fi

# Ensure docker-compose.yml exists before proceeding
if [ ! -f "docker-compose.yml" ]; then
    echo "docker-compose.yml file not found in the current directory."
    cleanup
fi

# Start Docker services in the background
docker-compose up -d || cleanup

# Check if Docker services are up and running
if ! docker-compose ps | grep "Up"; then
    cleanup
fi

# Run the create_bucket.py script
python create_bucket.py --access-key minioadmin --secret-key minioadmin --bucket-name metaflow || cleanup

echo "Setup completed successfully."
