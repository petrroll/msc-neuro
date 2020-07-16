out_name=$1; shift
err_name=$1; shift
file_name=$1; shift

cd /msc-neuro/
./env/bin/python3 "${file_name}" "$@" 1>>"$out_name" 2>>"$err_name"
