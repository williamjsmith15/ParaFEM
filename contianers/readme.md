# Instructions for how to build and use containers for ParaFEM

## Install Docker

To build and run the containers, you need to have Docker installed on your system. You can download Docker from the official website: [Docker Installation](https://docs.docker.com/get-docker/). Then follow the instructions for your operating system.

## Local Use

To build the container for local use, run the following command in the root directory of the ParaFEM repository:

```bash
docker build -t <image:tag> -f containers/Dockerfile .
```

then the contianer can be used with the following command:

```bash
docker run -it --rm \
    -v $(pwd):/home/ParaFEM \
    <image:tag> \
    /bin/bash
```

This should allow you to use any 

Explanation:

- `-it`: Runs the container in interactive mode with a terminal.
- `--rm`: Automatically removes the container when it exits.
- `-v $(pwd):/home/ParaFEM`: Mounts the current directory to `/home/ParaFEM` in the container, allowing you to access files from your host system.
- `<image:tag>`: The name and tag of the image you built. You can replace this with any name you prefer.
- `/bin/bash`: The command to run inside the container. This opens a bash shell, allowing you to interact with the container.

## Remote Use

To build the container for remote use, firstly sign up for a Docker Hub account and log in to your account using the command:

```bash
docker login
```

Then, to build the container for remote use, you need to replace `<uname/image:tag>` with your Docker Hub username and desired image name. To build the container, run the following command in the root directory of the ParaFEM repository:

```bash
docker build -t <uname/image:tag> -f containers/Dockerfile .
```

then the contianer can be uploaded to DockerHub with the following command:

```bash
docker push <uname/image:tag>
```

and ran on any system with Docker installed using the following command:

```bash
docker run -it --rm \
    -v $(pwd):/home/ParaFEM \
    <uname/image:tag> \
    /bin/bash
```

## Running on HPC Systems

As most HPC systems are more restricted than local machines, Docker no longer works as the containerisation service as it requires root access. Instead, Singularity is used as the containerisation service.

First install Singularity on your system. You can find the installation instructions here: [Singularity Installation](https://sylabs.io/guides/3.8/user-guide/installation.html).

To build the singularity container run:

```bash
sudo singularity build parafem.sif docker://<uname/image:tag>
```

then the .sif singularity contianer image can be transferred to the remote machine.

Load the singularity module on the remote machine and then run with the following command:

```bash
singularity exec --nv \
    -B $(pwd):/home/ParaFEM \
    parafem.sif \
    /bin/bash
```
