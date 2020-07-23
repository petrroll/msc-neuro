import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_mult1"
    exp = "bs3_exp9"
    exp_rev = "1"
    runner = urun.get_runner(sys.argv[1])
    run = 0

    for regions in [[1], [2], [3]]:
        for l2 in [0.0, 0.1, 1.0]:
            runner(
                exp_folder, exp,
                f"--exp_folder={exp_folder} --exp={exp}{exp_rev} --run={run} --hidden={0.2} --dog_layers={9} --regions={str(regions).replace(' ','')} --l2={l2}", 
                run
                )
            run += 1

    for hidden in [0.2, 0.3]:
        for dog_layer in [9, 15, 25]:
            for regions in [[1, 2], [1, 2, 3]]:
                for l2 in [0.0, 0.1, 1.0]:
                    runner(
                        exp_folder, exp,
                        f"--exp_folder={exp_folder} --exp={exp}{exp_rev} --run={run} --hidden={hidden} --dog_layers={dog_layer} --regions={str(regions).replace(' ','')} --l2={l2}", 
                        run
                        )
                    run += 1
