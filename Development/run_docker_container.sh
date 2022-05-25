docker build -t mentalhealth_development .

echo $PWD
docker run --rm -it \
    --runtime=nvidia \
    -w /workspace \
    -v $PWD/development_files:/workspace \
    -p 8888:8888 \
    --name mentalHealthDockerDevelopment \
    mentalhealth_development bash