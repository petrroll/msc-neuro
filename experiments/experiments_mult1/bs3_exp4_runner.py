import os
import sys

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_mult1"
    exp = "bs3_exp4"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    runner(
        exp_folder, exp,
        f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={0.2} --dog_layers={9} --regions={str([1]).replace(' ','')}", 
        run
        )
    run += 1
    runner(
        exp_folder, exp,
        f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={0.2} --dog_layers={9} --regions={str([2]).replace(' ','')}", 
        run
        )
    run += 1

    for hidden in [0.2, 0.4]:
        for dog_layer in [9, 15, 25]:
            for regions in [[1, 2], [1, 2, 3]]:
                runner(
                    exp_folder, exp,
                    f"--exp_folder={exp_folder} --exp={exp} --run={run} --hidden={hidden} --dog_layers={dog_layer} --regions={str(regions).replace(' ','')}", 
                    run
                    )
                run += 1
