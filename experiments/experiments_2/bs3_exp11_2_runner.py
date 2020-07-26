import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

#
# Dropout is `keep_prob` probability not `dropout rate` (as in TF2.x)
#

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs3_exp11"
    exp_rev = "2"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for hidden in [0.2, 0.3, 0.4]:
        for dropout_h in [0, 0.05, 0.1, 0.2, 0.4, 0.5]:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp}_{exp_rev} --run={run} --hidden={hidden} --dropout_h={1-dropout_h}", 
                f"{run}_{exp_rev}"
                )
            run += 1
