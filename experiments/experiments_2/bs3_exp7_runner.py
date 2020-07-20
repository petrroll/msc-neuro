import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs3_exp7"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for hidden in [0.1, 0.2, 0.4]:
        for c_filters in [9]:
            for c_size in [3, 5, 7, 15]:
                runner(
                    exp_folder, exp,
                    f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={hidden} --c_filters={c_filters} --c_size={c_size}", 
                    run
                    )
                run += 1
