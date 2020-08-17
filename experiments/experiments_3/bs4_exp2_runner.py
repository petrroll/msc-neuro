import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_3"
    exp = "bs4_exp2"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for reg_l in [0.0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 10_000, 100_000]:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp} --run={run} --reg_l={reg_l}", 
                run
                )
            run += 1
