import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_3"
    exp = "bs4_exp3"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    runner(
        exp_folder, exp,
        f"--exp_folder={exp_folder} --exp={exp} --run={run}", 
        run
        )
    run += 1
