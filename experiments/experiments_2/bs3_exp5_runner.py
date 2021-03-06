import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs3_exp5"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for hidden in [0.2, 0.4]:
        for reg_type in ['l1', 'l2']:
            for reg_lambda in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]:
                runner(
                    exp_folder, exp,
                    f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={hidden} --reg_lambda={reg_lambda} --reg_type={reg_type}", 
                    run
                    )
                run += 1
