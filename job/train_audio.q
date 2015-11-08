#!/bin/bash
# This line tells the shell how to execute this script, and is unrelated
# to PBS. In fact, by default PBS executes scripts with bash so this line
# could be omitted
  
# at the beginning of the script, lines beginning with "#PBS" are read by
# Torque and used to set queueing options. You can comment out a PBS
# directive with a second leading #, eg:
##PBS -l nodes=2:ppn=4
  
# we need 1 node, with 1 process per node: 
#PBS -l nodes=1:ppn=1
  
# we expect the job to finish within 5 hours. If it takes longer than 5
# hours, Torque can kill it:
#PBS -l walltime=1:00:00
  
# we expect the job to use no more than 2GB of memory:
#PBS -l mem=10GB
  
# we want the job to be named "jobname" rather than something generated
# from the script name. This will affect the name of the files where
# stdout and stderr are placed, and also the name of the job as reported
# by qstat:
#PBS -N test8
  
# if the job fails, send me an email at this email address.
#PBS -M arg450@nyu.edu
  
# instead of separate files for stdout and stderr, merge both into the
# stdout file. It will be placed in the directory I submitted the job
# from and will have a name like jobname.o12345
#PBS -j oe
  
# once the first non-comment, non-PBS-directive line is encountered, Torque
# stops looking for PBS directives. The remainder of the script is  executed
# as a normal Unix shell script
 
# first we ensure a clean running environment:
module purge
# and load the module for the software we are using:
module load numpy/intel/1.9.2
module load scikit-learn/intel/0.16.1
module load pandas/intel/0.16.2
module load librosa/intel/0.41
module load mir_eval/inte/0.1
 
# next we create a unique directory to run this job in. We will record its
# name in the shell variable "RUNDIR", for better readability.
# Torque sets PBS_JOBID to the job id, something like 12345.crunch.local
# ${PBS_JOBID/.*} expands to the job id up to the first '.'
# We take the job id (the '12345' in 'jobname.o12345')
# We make the run directory in our area under $SCRATCH, because at NYU HPC
# $SCRATCH is configured for the disk space and speed required by HPC jobs.
RUNDIR=$SCRATCH/my_project/run-${PBS_JOBID/.*}
mkdir $RUNDIR
 
# we will be reading data in from somewhere, so define that too:
DATADIR=$SCRATCH/my_project/data
 
# the script will have started running in $HOME, so we need to move into the 
# unique directory we just created
cd $RUNDIR
 
# now start the Stata job:
stata -b do $DATADIR/data_0706.do
 
# leave a blank line at the end, because Torque can "lose" the final line of the script otherwise