#!/bin/bash

#
# A example workflow MC->RECO->AOD for a simple pp min bias 
# production
export O2DPG_ROOT=/home/fmazzasc/alice/O2DPG

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

# ----------- LOAD UTILITY FUNCTIONS --------------------------
. ${O2_ROOT}/share/scripts/jobutils.sh

# ----------- START ACTUAL JOB  ----------------------------- 

NWORKERS=50
MODULES="--skipModules ZDC"
SIMENGINE=${SIMENGINE:-TGeant4}

# create workflow
${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py -eCM 14000  -col pp -gen pythia8 -proc inel -tf 10     \
                                                       -ns 100 -e ${SIMENGINE}                   \
                                                       -j ${NWORKERS} -interactionRate 500000    \
                                                       -confKey "Diamond.width[2]=6."            \
                                                       -mod "--skipModules ZDC"                  

${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json --cpu-limit 64

# publish the current dir to ALIEN
# copy_ALIEN
