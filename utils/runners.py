import os
import fire

def prepare_paths(exp_folder, exp):
    exp_path = f"./experiments/{exp_folder}"
    logs_folder = f"./job_logs/{exp_folder}/{exp}"
    os.makedirs(logs_folder, exist_ok=True)

    return exp_path, logs_folder

def run_qsub_cpu(exp_folder, exp, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    # CPU: qsub -q cpu.q -cwd -pe smp 4 -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G
    exp_path, logs_folder = prepare_paths(exp_folder, exp)
    os.system(
        f"qsub -q cpu.q -cwd -pe smp 4 -l mem_free=8G,act_mem_free=8G,h_data=20G \
        -o {logs_folder}/o_{run}.log \
        -e {logs_folder}/e_{run}.log \
            ./utils/run_in_env_with_cuda_aic.sh {exp_path}/{exp}.py {exp_args}")

def run_qsub_gpu(exp_folder, exp, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    # GPU: qsub -q gpu.q -cwd -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G 
    exp_path, logs_folder = prepare_paths(exp_folder, exp)
    os.system(
        f"qsub -q cpu.q -cwd -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G \
        -o {logs_folder}/o_{run}.log \
        -e {logs_folder}/e_{run}.log \
            ./utils/run_in_env_with_cuda_aic.sh {exp_path}/{exp}.py {exp_args}")

def run_env_win(exp_folder, exp, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    exp_path, logs_folder = prepare_paths(exp_folder, exp)
    exp_path, exp = exp_path.replace('/', '\\'), exp.replace('/', '\\') 
    os.system(
        f"start /B env\scripts\python.exe {exp_path}\\{exp}.py {exp_args} \
            >> {logs_folder}/o_{run}.log\
            2>> {logs_folder}/e_{run}.log")

def run_docker_cgg(exp_folder, exp, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    exp_path, logs_folder = prepare_paths(exp_folder, exp)
    os.system("docker container run -d --gpus 1 --mount \
        type=bind,source="+os.getcwd()+",target=/msc-neuro/ \
        houska/mscneuro "+
        f"bash /msc-neuro/utils/run_in_env_docker_cgg.sh \
            {logs_folder}/o_{run}.log \
            {logs_folder}/e_{run}.log \
            {exp_path}/{exp}.py {exp_args}")

def get_runner(environment):
    '''
    Gets a runner method based on environment.
    '''
    return {
        "qsub-cpu": run_qsub_cpu,
        "qsub-gpu": run_qsub_gpu,
        "env-win": run_env_win,
        "docker-cgg": run_docker_cgg,
    }[environment]

def generic_run(exp_folder, exp, runner):
    runner = get_runner(runner)
    runner(
            exp_folder, exp,
            f"--exp_folder={exp_folder} --exp={exp}", 
            run=0
            )

if __name__ == "__main__":
    fire.Fire(generic_run)