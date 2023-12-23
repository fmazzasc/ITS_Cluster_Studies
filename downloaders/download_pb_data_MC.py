import os
import yaml


output_dir = '/data/shared/ITS/OCT/MC/NEW/'
run_number_list = [505548]

for run_number in run_number_list:  
        if not os.path.exists(f'{output_dir}{run_number}'):
                os.makedirs(f'{output_dir}{run_number}')

        os.system(f'rm -rf {output_dir}{run_number}/*')
        os.system(f'alien.py find /alice/sim/2022/LHC22f1a3/{run_number}/0*/tf*/* > {output_dir}{run_number}/out.txt')

        with open(f'{output_dir}/{run_number}/out.txt') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]


        for index,line in enumerate(lines):
            print(line)
            if 'tf' in line and 'QC' not in line:
                os.system(f' alien.py cp -T 32 {line} file:{output_dir}/{line}')


