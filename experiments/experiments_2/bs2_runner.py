import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs2"
    runner = urun.get_runner(sys.argv[1])
    runner(
            exp_folder, exp,
            f"--exp_folder={exp_folder} --exp={exp}", 
            run=0
            )
