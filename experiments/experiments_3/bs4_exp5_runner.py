import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_3"
    exp = "bs4_exp5"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for hidden_lt in ['sep']:
        for c_size in [3,5,15]:
            for c_filters in [9, 25]:
                for cd2x in [0.1, 1]:
                    for hidden_t in ['l1', 'l2']: # l1 as described in the original paper
                        for hidden_s in [0.1,1]:
                            runner(
                                exp_folder, exp,
                                f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_size={c_size} --c_filters={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt={hidden_lt}", 
                                run
                                )
                            run += 1

    run += 100
    for hidden_lt in ['normal']:
        for c_size in [3,7,15]:
            for c_filters in [9, 25]:
                for cd2x in [0.1, 1]:
                    for hidden_t in ['max', 'l2']:
                        for hidden_s in [0.1, 1]:
                            runner(
                                exp_folder, exp,
                                f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_size={c_size} --c_filters={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt={hidden_lt}", 
                                run
                                )
                            run += 1
