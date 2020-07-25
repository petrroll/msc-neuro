import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs3_exp11"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for hidden in [0.2, 0.3, 0.4]:
        for dropout_h in [0, 0.05, 0.1, 0.2, 0.4, 0.5]:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={hidden} --dropout_h={dropout_h}", 
                run
                )
            run += 1
