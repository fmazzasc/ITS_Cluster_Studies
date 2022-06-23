#!/bin/bash

#
# A example workflow MC->RECO->AOD for a simple pp min bias 
# production
export O2DPG_ROOT=/home/spolitan/alice/O2DPG
export O2_ROOT=/home/spolitan/alice/AliceO2

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

# ----------- START ACTUAL JOB  ----------------------------- 

DISABLE_ROOT_OUTPUT=0 WORKFLOW_DETECTORS="ITS,TPC,TRD,TOF" IGNORE_EXISTING_SHMFILES=1  $O2_ROOT/prodtests/full-system-test/run-workflow-on-inputlist.sh CTF p_517619.dat 

#DISABLE_ROOT_OUTPUT=0 WORKFLOW_DETECTORS="ITS,TPC,TRD,TOF" $O2_ROOT/prodtests/full-system-test/run-workflow-on-inputlist.sh  TF <rawTFdata >

# publish the current dir to ALIEN
# copy_ALIEN
