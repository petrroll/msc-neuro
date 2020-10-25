import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_multeval"
    exps = ["bl2", "bl4", "bs4_exp5", "bs4_exp10", "bs4_exp1"]
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for exp in exps:
        for region in [1, 2, 3]:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp} --region={region}", 
                run
                )
            run += 1
