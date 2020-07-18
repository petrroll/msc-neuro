import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs3_exp2"
    runner = urun.get_runner(sys.argv[1])
    for run, bs in enumerate([2, 8, 16, 32, 64, 128, 512, 1024, 1800]):
        runner(
            exp_folder, exp,
            f"--exp_folder={exp_folder} --exp={exp} --run={run} --bs={bs}", 
            run
            )
