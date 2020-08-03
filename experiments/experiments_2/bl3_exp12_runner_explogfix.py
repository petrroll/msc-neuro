run = 0
for hidden in [0.2, 0.3]:
    for reg_h in [0, 0.05, 0.1, 0.5]:
        for reg_l in [0, 0.05, 0.1, 0.5]:
            print(f"experiments_2/bl3_exp12x{run}/baseline3_l2xH{reg_h}xL{reg_l}_N{hidden}x35000")
            run += 1