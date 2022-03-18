import os
import yaml

run_number = 505673
if not os.path.exists(f'{run_number}'):
        os.makedirs(f'{run_number}')

os.system(f'rm -rf {run_number}/*')
os.system(f'alien.py ls /alice/data/2021/OCT/{run_number}/apass3/ > {run_number}/out.txt')

with open(f'{run_number}/out.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

input_dir = f'/alice/data/2021/OCT/{run_number}/apass3/'

for index,line in enumerate(lines):
    if line[0:6] == 'o2_ctf':
        input_path = input_dir + line
        if not os.path.exists(line):
                os.makedirs(line)
        os.system(f' alien.py cp -T 32 {input_path}tpctracks.root file:{run_number}/{line}tpc_tracks.root')
        os.system(f' alien.py cp -T 32 {input_path}o2trac_its.root file:{run_number}/{line}o2trac_its.root')
        os.system(f' alien.py cp -T 32 {input_path}o2clus_its.root file:{run_number}/{line}o2clus_its.root')
        os.system(f' alien.py cp -T 32 {input_path}o2_primary_vertex.root file:{run_number}/{line}o2_primary_vertex.root')



# os.system(f'hadd {run_number}/o2match_itstpc.root part_merging/o2match_itstpc.root*')
# os.system(f'hadd {run_number}/o2clus_its.root part_merging/o2clus_its.root*')
# os.system(f'hadd {run_number}/o2trac_its.root part_merging/o2trac_its.root*')
# os.system(f'hadd {run_number}/o2_primary_vertex.root part_merging/o2_primary_vertex*')


# os.system('rm -rf part_merging')
# os.system('rm -rf out.txt')

