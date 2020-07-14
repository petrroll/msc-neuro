FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN apt-get update
RUN apt-get install -y build-essential

# https://stackoverflow.com/questions/39539110/pyvenv-not-working-because-ensurepip-is-not-available
RUN apt-get install -y python3-pip python3-setuptools python3.7-dev python3.7-venv
EXPOSE 6006 8888

#
# Note:
# I know that I should bake dependencies into the docker image, copy all relevant files in (./Data, ./experiments, ./NDN, ...), 
# ..and only ever mount ./logs and ./models but using a container as a persistent VM is just more convinient to me atm for this 
# ..very particular usecase. 
#
# Note 2: 
# - The commented-out approach uses virtenv even though it doesn't make sense in container environemnt to have 100 % consistent 
# ..setup with non-docker environments. I'm fully aware that it's not ideal either and docker-idiomatic solution would look
# ..totally different. But if it works...
#
# WORKDIR /msc-neuro
# COPY Makefile requirements.txt requirements-dev.txt NDN3/requirements.txt NDN3/requirements-gpu.txt NDN3/requirements-core.txt ./
# RUN make env-dev
# COPY ./ ./
#
# Note 3:
# - Run jupyter: `jupyter notebook --ip=0.0.0.0 --allow-root`
# - Run docker: `docker container run -p 36006:6006 -p 38888:8888 -it --gpus 1 --mount type=bind,source="$(pwd)",target=/msc-neuro/ houska/mscneuro`
#