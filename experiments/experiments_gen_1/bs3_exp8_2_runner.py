import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_gen_1"
    exp = "bs3_exp8"
    exp_rev = "2"
    runner = urun.get_runner(sys.argv[1])
    run = 0

    # `1.3` seems to produce the same correlation between `output_tr_gen` (our assumed truth) & `output_tr_gen_noise` (what we'll train against) 
    # .. as is the correlation between `output_tr_gen` (best model) and `output_tr` (gold data)
    for noise_coef in [0.3, 0.35, 0.4]:
        for eval_noised in [True, False]:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp}_{exp_rev} --run={run} --noise_coef={noise_coef} --eval_noised={eval_noised}", 
                run
                )
            run += 1
