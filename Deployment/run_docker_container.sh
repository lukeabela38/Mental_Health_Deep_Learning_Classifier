docker build -t mentalhealth_deployment .

echo $PWD
docker run --rm -it \
    --runtime=nvidia \
    -w /workspace \
    -v $PWD/deployment_files:/workspace \
    -p 8000:8000 \
    -p 8001:8001 \
    --name mentalHealthDockerDeployment \
    mentalhealth_deployment bash