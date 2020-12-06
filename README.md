# Blub

Experimenting with GPU driven 3D fluid simulation on the GPU using [WebGPU-rs](https://github.com/gfx-rs/wgpu-rs).  
Focusing primarily on hybrid approaches lagrangian/eularian approaches here (PIC/FLIP/APIC..).

For SPH (pure lagrangian) fluid simulation, check out my simple 2D DFSPH fluid simulator, [YASPH2D](https://github.com/Wumpf/yasph2d).

[![](https://img.youtube.com/vi/b8_HMJhM7Ws/hqdefault.jpg)](https://www.youtube.com/watch?v=b8_HMJhM7Ws "Blub fluid simulation video")

## Application / Framework

### Build & Run

Requires git-lfs (for large textures & meshes).

`cargo run`
Note that there are a few extra dependencies due to the `shaderc`, if your build fails check shaderc-rs' [build instructions](https://github.com/google/shaderc-rs#building-from-source).  
Should work on Linux/Windows - I'm developing on Windows, so things might break at random for the others.  
**Mac is not working right now #26**.

Doing release mode (`cargo run --release`) can be significantly faster.

First time loading any scene/background is a bit slower since some of the pre-computations are cached on disk. In particular:
* raw cubemap texture (decoding the .hdr takes surprisingly long)
* computing signed distance field (happens brute force on gpu)

### Shaders

GLSL, compiled to SPIR-V at runtime. Shaders are hot reloaded on change, have fun!  
(on failure it will keep using the previously loaded shader)

### "Scenes"

Simple json format where I dump various properties that I think are either too hard/annoying to set via UI at all or I'd like to have saved.
Can be reloaded at runtime and will pick up any change

### Major Dependencies

* [WebGPU-rs](https://github.com/gfx-rs/wgpu)
  * [webgpu](https://gpuweb.github.io/gpuweb/) but in Rust!
  * as of writing all this is still in heavy development, so I'm using some master version, updated in irregular intervals
* [DearImGUI](https://github.com/ocornut/imgui)
  * or rather, its [Rust binding](https://github.com/Gekkio/imgui-rs)
  * I'm maintaining a fork of the webgpu-rs binding layer [here](https://github.com/Wumpf/imgui-wgpu-rs/tree/use-wgpu-master) to be able to use newest version
* various other amazing crates, check [cargo.toml](https://github.com/Wumpf/blub/blob/master/Cargo.toml) file

## Simulation

To learn more about fluid simulation in general, check out [my Gist on CFD](https://gist.github.com/Wumpf/b3e953984de8b0efdf2c65e827a1ccc3) where I gathered a lot of resources on the topic.

Implements APIC, [SIGGRAPH 2015, Jiang et al., The Affine Particle-In-Cell Method](https://www.math.ucla.edu/%7Ejteran/papers/JSSTS15.pdf) and [IEEE Transactions on Visualization and Computer Graphics 2019, Kugelstadt et al., Implicit Density Projection for Volume Conserving Liquids](https://animation.rwth-aachen.de/media/papers/66/2019-TVCG-ImplicitDensityProjection.pdf) on GPU

Noted down a few interesting implementation details here.

### Particle to Grid Transfer

Transferring the particle's velocity to the grid is tricky & costly to do in parallel!
Either, velocities are scattered by doing 8 atomic adds for every particle to surrounding grid cells, or grid cells traverse all neighboring cells. (times 3 for staggered grid!)
There's some very clever ideas on how to do efficient scattering in [Ming et al 2018, GPU Optimization of Material Point Methods](http://pages.cs.wisc.edu/~sifakis/papers/GPU_MPM.pdf) using subgroup operations (i.e. inter warp/wavefront shuffles) and atomics.
Note though that today atomic floats addition is pretty much only available in CUDA and OpenGL (using an NV extension) and subgroup operations are not available in wgpu as of writing.

In Blub I tried a (to my knowledge) new variant of the gather approach:
Particles form a linked list by putting their index with a atomic exchange operation in a "linked list head pointer grid" which is a grid dual to the main velocity volume.
Then, every velocity grid cell walks 8 neighboring linked lists to gather velocity.
(this makes this a sort of hybrid between naive scatter and gather)

Note that this all makes MAC/staggered grids a lot less appealing since the volume in which particles need to be accumulated gets bigger & more complicated, i.e. a lot slower.
After various tries with collocated grids I ended up using staggered after all (for some details see #14) since I couldn't get the collocated case quite right.
(to avoid artifacts with collocated grids, Rhie-Chow interpolation is required. It's widespread in CFD since collocated grids are required for arbitrary meshes, but it's hard to find any resources in the computer graphics community [...])  
Generally, one can either stick with a single linked list grid (sampling 6 different but overlapping cells per velocity component) or three different linked-list grids.
I eventually settled with three different grids, processing a single velocity component at a time.

Note that _by far_ the biggest bottleneck in this approach is walking the particle linked list. Doing a shared memory optimization yielded >4x performance speed up:
Every thread walks only a single linked list, stores the result to shared memory and then reads the remaining seven neighbor linked lists from shared memory. 👌

### Solver

Using Preconditioned Conjugate Gradient solver for solving the poisson pressure equation (PPE). In comments and naming in the code I'm following the description in [Bridson's book](https://www.amazon.com/Simulation-Computer-Graphics-Robert-Bridson/dp/1568813260).
Implementing it in compute shader isn't entirely straight forward and needs some optimizing.
Blub is using an [Incomplete Poisson](https://software.intel.com/content/www/us/en/develop/articles/parallelized-incomplete-poisson-preconditioner-in-cloth-simulation.html) Preconditioner, better and shorter described by [Austin Eng here](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid#pressure-solve).

I started out with Jacobi iterations - very easy to implement, but inaccurate and slow (many iterations necessary). This is a good starting point though if you implement your own solver - [here's](https://github.com/Wumpf/blub/blob/c02ea18/shader/simulation/pressure_solve.comp) what the code looked like.

Looked into [A Multigrid Fluid Pressure SolverHandling Separating Solid Boundary Conditions, Chentanez et al. 2011](https://matthias-research.github.io/pages/publications/separatingBoundaries.pdf)
for a while but shied away from implementing such a complex solver at the moment without any reference code and with too little personal experience in the field.

#### Iteration Control

Typically solvers are run until a certain error threshold is reached. This is notoriously tricky on GPU, since this means that we need to have the mean squared error (MSE) feed back to determine how many more dispatch calls for solver iterations should be issued. We can't wait for the result as this would introduce a GPU-CPU stall. Experimenting with using MSE from several iterations ago (i.e. asynchronously querying the MSE) didn't yield promising results due to strong fluctuations and varying delay. Blub follows a different strategy instead:

There is a fix maximum number of iterations which determines how many compute dispatches are issued (note that there are several per iteration!), however most of these dispatches are indirect, so when evaluating the MSE, we may null out the indirect dispatch struct, making the remaining dispatches rather cheap (still not free though!).
Since evaluating the MSE itself is costly, this is done every couple of few iterations (configurable).

The last computed MSE and iteration count is queried asynchronously, in order to display a histogram in the gui and make informed choices for selecting the target MSE, max iteration & MSE evaluation frequency parameters.

#### Implicit Density Projection

For improved volume conversation & iteration times Blub implements a "secondary pressure solver" that uses fluid density instead of divergence as input. A video + paper can be found [here](https://animation.rwth-aachen.de/publication/0566/). I found that it improves the quality of the simulation tremendously for large timesteps (I typically run the simulation/solver at 120hz).

Compared to what is described (to my understanding) in the paper I made a few adjustments/trade-offs:
* For computing densities, neighboring solid cells are assumed to have a fixed (interpolation kernel derived) density contribution instead of sampling it with particles
* No resampling for degenerated cases - they are rather hard to detect and handle on GPU

## Rendering

Particle visualization with quads. Put a ridiculous amount of effort into to make the quads display perspective correct spheres.

The basic idea of screen space fluid rendering is very well described in these [GDC 2010 slides](http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf).
The implementation here is driven by a the depth filer described in [A Narrow-Range Filter for Screen-Space Fluid Rendering, Truong et al. 2018](http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf) which I tried to make reasonably efficient with some shared memory optimizations.  
On top of that comes some hand wavy (pun unintended) physically based rendering things, best check the comments in the shader code if you want to learn more ;-).

## Trivia

### Name
From German *[blubbern](https://en.wiktionary.org/wiki/blubbern)*, to bubble.  
Found out later that there was a [water park in Berlin](https://en.wikipedia.org/wiki/Blub_(water_park)) with that name, but it closed down in 2002.
