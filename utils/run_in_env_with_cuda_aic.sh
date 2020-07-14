LD_LIBRARY_PATH="/lnet/aic/opt/cuda/cuda-10.0/lib64"
file_name=$1; shift
./env/bin/python3 ${file_name} "$@"
