import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_3"
    exp = "bs4_exp6"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for c_filters in [12, 24, 48]:
        for cd2x in [0.1, 1]:
            for norm2_filt in [0.1, 1]:
                for l1 in [0.1, 1]:
                    runner(
                        exp_folder, exp,
                        f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_filters={c_filters} --cd2x={cd2x} --norm2_filt={norm2_filt} --l1={l1}", 
                        run
                        )
                    run += 1
