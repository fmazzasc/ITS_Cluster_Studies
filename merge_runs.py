import os
import yaml

run_number = 505658

os.system(f'alien.py ls /alice/data/2021/OCT/{run_number}/apass2/ > out.txt')




with open('out.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

if not os.path.exists(f'{run_number}'):
        os.makedirs(f'{run_number}')

if not os.path.exists('part_merging'):
        os.makedirs('part_merging')


input_dir = f'alien:/alice/data/2021/OCT/{run_number}/apass2/'

for index,line in enumerate(lines):
    if(index>2): 
            continue
    if line[0:6] == 'o2_ctf':
        input_path = input_dir + line
        os.system(f' alien_cp -T 32 {input_path}o2match_itstpc.root part_merging/o2match_itstpc.root{index}')
        os.system(f' alien_cp -T 32 {input_path}o2trac_its.root part_merging/o2trac_its.root{index}')
        os.system(f' alien_cp -T 32 {input_path}o2clus_its.root part_merging/o2clus_its.root{index}')


os.system(f'hadd {run_number}/o2match_itstpc.root part_merging/o2match_itstpc.root*')
os.system(f'hadd {run_number}/o2clus_its.root part_merging/o2clus_its.root*')
os.system(f'hadd {run_number}/o2trac_its.root part_merging/o2trac_its.root*')

os.system('rm -rf part_merging')
os.system('rm -rf out.txt')


