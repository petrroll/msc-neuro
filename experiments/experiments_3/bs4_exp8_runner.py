import os
import sys
from sys import last_type

sys.path.append(os.getcwd())

import utils.runners as urun

if __name__ == "__main__":
    exp_folder = "experiments_3"
    exp = "bs4_exp8"
    runner = urun.get_runner(sys.argv[1])
    run = 0
    for c_filters in [48]:
        for cd2x in [0.1, 0.5, 1]:
            for conv_reg_type in ["max_filt", "l2", "max"]:
                for conv_reg_str in [1, 10]:
                    for last_type in ["l1", "l2"]:
                        for last_str in [0.005, 0.01, 0.05] if last_type == "l1" else [0.01, 0.05, 0.1]:
                            for secondary_conv_size in [3]:
                                for spacing in [2]:
                                    runner(
                                        exp_folder, exp,
                                        f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_filters={c_filters} --cd2x={cd2x} --conv_reg_str={conv_reg_str} --conv_reg_type={conv_reg_type} --last_type={last_type} --last_str={last_str} --secondary_conv_size={secondary_conv_size} --spacing={spacing}", 
                                        run
                                        )
                                    run += 1

    run = 100
    for c_filters in [48]:
        for cd2x in [0.1]:
            for conv_reg_type in ["max_filt", "l2", "max"]:
                for conv_reg_str in [0.01, 0.1, 1]:                                    
                    for last_type in ["l1"]:
                        for last_str in [0.01]:
                            for secondary_conv_size in [3, 8]:
                                for spacing in [1, 2, 0]:
                                    runner(
                                        exp_folder, exp,
                                        f"--exp_folder={exp_folder} --exp={exp} --run={run} --c_filters={c_filters} --cd2x={cd2x} --conv_reg_str={conv_reg_str} --conv_reg_type={conv_reg_type} --last_type={last_type} --last_str={last_str} --secondary_conv_size={secondary_conv_size} --spacing={spacing}", 
                                        run
                                        )
                                    run += 1
