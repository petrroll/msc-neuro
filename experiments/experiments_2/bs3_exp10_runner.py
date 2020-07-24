import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_2"
    exp = "bs3_exp10"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for lin_scale in [True, False]:
        for input_scale in ['normalize_mean_std', 'times_mil', 'identity']:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp} --run={run} --lin_scale={lin_scale} --input_scale={input_scale}", 
                run
                )
            run += 1
