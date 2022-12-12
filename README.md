# brain-projection-age
This repository contain a ML model for predicting brain age using only two dimensional images derived from MR brain volumes. The two dimensional images correspond
to different viewing axes (think x, y & z) which will be referred to as projections. All used such images (for each brain volume) will be called channels and each
channel belongs to exactly one projection.

The basic case is to feed the model two channels for each projection, the mean intensity along the axis of projection and the standard deviation of same. In my research
I have trained variants of this model on structural T1 brain volumes from UK biobank.

## Usage
As this has very much been code in flux (and `Flux` as it happens), I foresee that it might not be trivial to run. Nevertheless I have tried to clean up the code
and provide comments in markup as well as inside the code itself and will below attempt to explain how it's meant to be used and what assumptions have been made.

The data is assumed to be volumes of size (or padded to) 208⨯256⨯256 voxels contained in (possibly gzip-compressed) nifti files.

The notebook is meant to be served from a jupyter server (either notebook or lab), with a [Julia](https://julialang.org/downloads) [1.6 kernel](https://julialang.github.io/IJulia.jl/stable/manual/installation/#Installing-additional-Julia-kernels). It is likely that later versions of Julia will also run
the code. It will create a Julia environment to which all dependencies need to be added. The notebook will also attempt to do that, but in case it doesn't work, manual
intervention could be required.

Some adaptations to your computing environment also has to be made. Specifically, under the heading "Commandline Interface" a string of arguments such as is typically
given on the commandline to a shell command, can be entered in the first cell of this section. The second cell (or it's output) will provide an explanation. Furthermore,
under the heading "Hardcoded Paths" the paths of the directory with data and the csv file identifying each volume and listing age and gender (as 0 or 1) should be
adjusted. The directory with brain volumes should be writable, because all two dimensional images derived from the volumes, are cached in a hidden directory to obviate
unnecessary recalculation.

The section "Main procedure" is intended to be easy enough to follow for someone to try out modifications or to add extra cells to experiment in. Should one want to add
training sessions in a file for use with the `runfromfile` function, the syntax is explained below.

### Job format
Each line (that will run a job, otherwise comments are accepted and ignored) must conform to this structure:
```
lineno args-definition kwargs-definition
```
These three parts are separated by whitespace. Note that whitespace may also be allowed within the parts.

#### `lineno`
The `lineno` is an integer. It's currently not used (but still compulsory), but is intended to be used for such things as giving a determined gpu assignment for each job. Otherwise every process running `runfromfile` will always consume the first job in the file.
#### `args-definition`
This part is intended to give arguments to the `train_and_evaluate` function. As such it should be a valid Julia expression of the form:
```
args = (modelfile, modelarguments, modelmodifications, hyperparameters)
```
The `modelfile` argument is the filename where the julia code for the model resides. The `modelarguments` is a collection (e.g. tuple) of arguments for the model. The
of the models presently provided here only the `channel_toggle` models use this mechanism. They should be provided with a collection of numbers representing what 
channels should be used.

The `modelmodifications` is a tuple of functions that will be called 
with the model as the only argument, intended to make it possible to modify the model on the fly. These functions would typically be small anonymous functions that call
the helper function `changelayers`. Here is an example of such a function:
```
model -> changelayers(model, (Flux.Dropout,), ((:p => 0.7),))
```
The second argument to `changelayers` is a tuple of all the layertypes in the model that should be adressed. In this case only dropout layers. The third argument is a 
tuple of all the modifications to be made to said layers. These can be expressed as `Pair`s of valid fieldnames for the layers in question and values to assign to said 
field names. Alternatively, such a modifications can also be expressed as a function taking the layer to modify as its only argument. Note that one call to 
`changelayers` can apply several changes to layers of several types. For better granularity, several modifications, each calling `changelayers` could be given in the 
`modelmodifications` tuple.

The `hyperparameters` argument is a dictionary (could be empty, but must never-the-less be given) with hyperparameters and their values. 
Currently only `:lr` meaning learning rate is used. If not in the dictionary it defaults to 0.003.

Below is a complete `args-definition`:
```
                        model filename                  arguments for the model           modifications (in this case 1) to apply to model     hyperparameters
      ╭───────────────────────┴───────────────────────╮  ╭─────────┴─────────╮  ╭───────────────────────────────┴──────────────────────────────╮  ╭──┴─╮
args=("models/model_channel_toggle.with_do_in_dense.jl", ([1, 2, 3, 4, 5, 6],), (model -> changelayers(model, (Flux.Dropout,), ((:p => 0.7),)),), Dict())
                                                                                                              ╰──────┬──────╯  ╰──────┬─────╯
                                                                                                           layer types affected    change(s)
```
#### `kwargs-definition`
This is a dictionary of keyword arguments for the `train_and_evaluate`. This could be left empty or the key `:identifier` could be assigned a string with extra 
information to insert in filenames associated with this run. Note that said string should not violate any filename conventions by containing slashes, null-characters, 
wildcard characters et.c.
