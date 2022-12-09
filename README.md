# brain-projection-age
This repository contain a ML model for predicting brain age using only two dimensional images derived from MR brain volumes. The two dimensional images correspond
to different viewing axes (think x, y & z) which will be referred to as projections. All used such images (for each brain volume) will be called channels and each
channel belongs to exactly one projection.

The basic case is to feed the model two channels for each projection, the mean intensity along the axis of projection and the standard deviation of same. In my research
I have trained variants of this model on structural T1 brain volumes from UK biobank.

## Usage
As this has very much been code in flux (and Flux as it happens), I foresee that it might not be trivial to run. Nevertheless I have tried to clean up the code
and provide comments in markup as well as inside the code itself and will below attempt to explain how it's meant to be used and what assumptions have been made.

The notebook is meant to be served from a jupyter server (either notebook or lab), with a Julia 1.6 kernel. It is likely that later versions of Julia will also run
the code. It will create a Julia environment to which all dependencies need to be added. The notebook will also attempt to do that, but in case it doesn't work, manual
intervention could be required.

Some adaptations to your computing environment also has to be made. Specifically, under the heading "Commandline Interface" a string of arguments such as is typically
given on the commandline to a shell command, can be entered in the first cell of this section. The second cell (or it's output) will provide an explanation. Furthermore,
under the heading "Hardcoded Paths" the paths of the directory with data and the csv file identifying each volume and listing age and gender (as 0 or 1) should be
adjusted. The directory with brain volumes should be writable, because all two dimensional images derived from the volumes, are cached in a hidden directory to obviate
unnecessary recalculation.

The section "Main procedure" is intended to be easy enough to follow for someone to try out modifications or to add extra cells to experiment in. Should one want to add
training sessions in a file for use with the `runfromfile` function, the syntax is explained below.
