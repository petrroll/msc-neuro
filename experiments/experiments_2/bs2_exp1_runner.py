import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs2_exp1"
    runner = urun.get_runner(sys.argv[1])
    for run, lr in enumerate([1e-5, 1e-4, 5e-4, 1e-3, 5e-3]):
        runner(
            exp_folder, exp,
            f"--run={run} --learning_rate={lr}", 
            run
            )
