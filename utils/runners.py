import os

def run_qsub_cpu(exp_invocation, exp_folder, exp, run):
    '''
    Run an exp_invocation on aic qsub with 4 CPUs. Expects msc-neuro folder structure.
    '''
    # CPU: qsub -q cpu.q -cwd -pe smp 4 -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G
    exp_path = f"./experiments/{exp_folder}"
    logs_folder = f"./job_logs/{exp_folder}/{exp}"
    os.makedirs(logs_folder, exist_ok=True)

    os.system(
        f"qsub -q cpu.q -cwd -pe smp 4 -l mem_free=8G,act_mem_free=8G,h_data=20G \
        -o {logs_folder}/o_{run}.log \
        -e {logs_folder}/e_{run}.log \
            ./utils/run_in_env_with_cuda_aic.sh {exp_path}/{exp_invocation}")

def run_qsub_gpu(exp_invocation, exp_folder, exp, run):
    '''
    Run an exp_invocation on aic qsub with GPU. Expects msc-neuro folder structure.
    '''
    # GPU: qsub -q gpu.q -cwd -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G 
    exp_path = f"./experiments/{exp_folder}"
    logs_folder = f"./job_logs/{exp_folder}/{exp}"
    os.makedirs(logs_folder, exist_ok=True)

    os.system(
        f"qsub -q cpu.q -cwd -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G \
        -o {logs_folder}/o_{run}.log \
        -e {logs_folder}/e_{run}.log \
            ./utils/run_in_env_with_cuda_aic.sh {exp_path}/{exp_invocation}")


def get_runner(environment):
    '''
    Gets a runner method based on environment.
    '''
    return {
        "qsub-cpu": run_qsub_cpu,
        "qsub-gpu": run_qsub_gpu,
    }[environment]
