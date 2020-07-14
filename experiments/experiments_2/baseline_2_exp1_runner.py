import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp_base = "bs2_exp1"
    exp_file = "baseline_2_exp1.py"
    runner = urun.get_runner(sys.argv[1])
    for run, lr in enumerate([1e-5, 1e-4, 5e-4, 1e-3, 5e-3]):
        runner(
            f"{exp_file} --run={run} --learning_rate={lr}", 
            exp_folder, f"{exp_base}x{run}", run
            )
