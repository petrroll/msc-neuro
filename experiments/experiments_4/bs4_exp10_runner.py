import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_4"
    exp = "bs4_exp10"
    runner = urun.get_runner(sys.argv[1])
    run = 200
    for hidden_lt in ['sep']:
        for c_size in [7,15]:
            for c_filters in [9]:
                for hidden_t in ['l1']: # l1 as described in the original paper
                    for hidden_s in [0.01, 0.1]:
                        runner(
                            exp_folder, exp,
                            f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_size={c_size} --c_filters={c_filters} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt={hidden_lt}", 
                            run
                            )
                        run += 1

    run += 100
    for hidden_lt in ['normal']:
        for c_size in [7,15,]:
            for c_filters in [9]:
                for hidden_t in ['l1']:
                    for hidden_s in [0.01, 0.1,]:
                        runner(
                            exp_folder, exp,
                            f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_size={c_size} --c_filters={c_filters} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt={hidden_lt}", 
                            run
                            )
                        run += 1
