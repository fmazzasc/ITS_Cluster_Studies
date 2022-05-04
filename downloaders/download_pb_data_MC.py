import os
import yaml

run_number_list = [301004]
target_dir = 'data/fmazzasc/its_data/MC'

for run_number in run_number_list:  
        if not os.path.exists(f'{run_number}'):
                os.makedirs(f'{run_number}')

        # os.system(f'rm -rf {run_number}/*')
        os.system(f'alien.py find /alice/sim/2021/LHC21i1a4/{run_number}/0*/tf*/o2match_tof_itstpc.root > {run_number}/out.txt')

        with open(f'{run_number}/out.txt', 'rb') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        for index,line in enumerate(lines):
            line = line.decode('utf-8')
            if 'tf' in line and 'QC' not in line:
                line = line.replace('o2match_tof_itstpc.root', '')
                print(line)
                if not os.path.exists(f'{run_number}/' + f'o2_ctf_{index}'):
                        os.makedirs(f'{run_number}/' + f'o2_ctf_{index}')
                print("--------------------------------------------------")
                print(f"CTF {index} / {len(lines)} ")
                os.system(f' alien.py cp -T 32 {line}/o2match_itstpc.root file:{target_dir}/{run_number}/o2_ctf_{index}/o2match_itstpc.root')
                os.system(f' alien.py cp -T 32 {line}/tpctracks.root file:{target_dir}/{run_number}/ctf_{index}/tpctracks.root')
                os.system(f' alien.py cp -T 32 {line}/o2trac_its.root file:{target_dir}/{run_number}/ctf_{index}/o2trac_its.root')
                os.system(f' alien.py cp -T 32 {line}/o2clus_its.root file:{target_dir}/{run_number}/ctf_{index}/o2clus_its.root')
                os.system(f' alien.py cp -T 32 {line}/o2_primary_vertex.root file:{target_dir}/{run_number}/ctf_{index}/o2_primary_vertex.root')


