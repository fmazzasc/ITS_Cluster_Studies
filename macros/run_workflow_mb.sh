
#!/usr/bin/env bash

ln -nsf bkg_grp.root o2sim_grp.root
ln -nsf bkg_geometry.root o2sim_geometry.root

export O2DPG_ROOT=/home/fmazzasc/alice/O2DPG



# # ----------- LOAD UTILITY FUNCTIONS --------------------------
. ${O2_ROOT}/share/scripts/jobutils.sh 

RNDSEED=1995
NSIGEVENTS=200
NWORKERS=60
NTIMEFRAMES=10

${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py -e TGeant4 -eCM 13500 -gen pythia8 -proc inel -col pp -j ${NWORKERS} -ns ${NSIGEVENTS} -tf ${NTIMEFRAMES} -mod "--skipModules ZDC" -field 5 \
	-confKey "Diamond.width[2]=6.;Diamond.width[0]=0.01;Diamond.width[1]=0.01;MaterialManagerParam.inputFile=/data/fmazzasc/its_data/sim/MB/medium_params.json"

# run workflow
# allow increased timeframe parallelism with --cpu-limit 32 
${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt aod --cpu-limit 64