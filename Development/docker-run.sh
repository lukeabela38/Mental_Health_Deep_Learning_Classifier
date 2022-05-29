docker build -t mentalhealth_development .

docker run --rm -it \
    --runtime=nvidia \
    -w /workspace \
    -p 8888:8888 \
    -v $PWD/development_files:/workspace \
    --name mentalHealthDockerDevelopment \
    mentalhealth_development bash