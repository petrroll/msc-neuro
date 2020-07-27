import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bl3_exp12"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for hidden in [0.2, 0.3]:
        for reg_h in [0, 0.05, 0.1, 0.5]:
            for reg_l in [0, 0.05, 0.1, 0.5]:
                    runner(
                        exp_folder, exp,
                        f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={hidden} --reg_h={reg_h} --reg_l={reg_l}", 
                        run
                        )
                    run += 1
