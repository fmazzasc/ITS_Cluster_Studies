import os
import yaml

run_number = 505582
target_dir = '/data/fmazzasc/its_data/PBdata/BNEG'
input_dir = f'/alice/data/2021/OCT/{run_number}/apass3/'


if not os.path.exists(f'{target_dir}/{run_number}'):
        os.makedirs(f'{target_dir}/{run_number}')

os.system(f'rm -rf {target_dir}/{run_number}/*')
os.system(f'alien.py ls /alice/data/2021/OCT/{run_number}/apass3/ > {target_dir}/{run_number}/out.txt')

with open(f'{target_dir}/{run_number}/out.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]


for index,line in enumerate(lines):
    if line[0:6] == 'o2_ctf':
        input_path = input_dir + line
        if not os.path.exists(line):
                os.makedirs(line)
        os.system(f' alien.py cp -T 32 {input_path}o2match_itstpc.root file:{target_dir}/{run_number}/{line}o2match_itstpc.root')
        os.system(f' alien.py cp -T 32 {input_path}tpctracks.root file:{target_dir}/{run_number}/{line}tpctracks.root')
        os.system(f' alien.py cp -T 32 {input_path}o2trac_its.root file:{target_dir}/{run_number}/{line}o2trac_its.root')
        os.system(f' alien.py cp -T 32 {input_path}o2clus_its.root file:{target_dir}/{run_number}/{line}o2clus_its.root')
        os.system(f' alien.py cp -T 32 {input_path}o2_primary_vertex.root file:{target_dir}/{run_number}/{line}o2_primary_vertex.root')

