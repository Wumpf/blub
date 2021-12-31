# Blub

Experimenting with GPU driven 3D fluid simulation on the GPU using [WebGPU-rs](https://github.com/gfx-rs/wgpu-rs).  
Focusing on hybrid lagrangian/eularian approaches here (PIC/FLIP/APIC..).

[![](https://img.youtube.com/vi/Y646KLHyIms/hqdefault.jpg)](https://www.youtube.com/watch?v=Y646KLHyIms "Blub fluid simulation video")

I gave a talk about it at the Rust Graphics Meetup:
* [Video](https://www.youtube.com/watch?v=Yzr9va5UtiE)
* [Slides](https://docs.google.com/presentation/d/105vjRxwaXVOhHWQJeaEZh3Nu-TtFWOwNAFroi0GFuzc/edit?usp=sharing)

For SPH (pure lagrangian) fluid simulation, check out my simple 2D DFSPH fluid simulator, [YASPH2D](https://github.com/Wumpf/yasph2d).

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
* [egui](https://github.com/emilk/egui)
* [winit](https://github.com/rust-windowing/winit)
* various other amazing crates, check [cargo.toml](https://github.com/Wumpf/blub/blob/master/Cargo.toml) file

## Simulation

To learn more about fluid simulation in general, check out [my Gist on CFD](https://gist.github.com/Wumpf/b3e953984de8b0efdf2c65e827a1ccc3) where I gathered a lot of resources on the topic.

Implements APIC, [SIGGRAPH 2015, Jiang et al., The Affine Particle-In-Cell Method](https://www.math.ucla.edu/%7Ejteran/papers/JSSTS15.pdf) and [IEEE Transactions on Visualization and Computer Graphics 2019, Kugelstadt et al., Implicit Density Projection for Volume Conserving Liquids](https://animation.rwth-aachen.de/media/papers/66/2019-TVCG-ImplicitDensityProjection.pdf) on GPU

Noted down a few interesting implementation details here.

### Particle to Grid Transfer

Transferring the particle's velocity to the grid is tricky & costly to do in parallel!
Either, velocities are scattered by doing 8 atomic adds for every particle to surrounding grid cells, or grid cells traverse all neighboring cells. (times 3 for staggered grid!)
There's some very clever ideas on how to do efficient scattering in [Ming et al 2018, GPU Optimization of Material Point Methods](http://pages.cs.wisc.edu/~sifakis/papers/GPU_MPM.pdf) using subgroup operations (i.e. inter warp/wavefront shuffles) and atomics.
Note though that today atomic floats addition is pretty much only available in CUDA and OpenGL/[Vulkan](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VK_EXT_shader_atomic_float) (using extensions which are only supported on Nvidia) and subgroup operations are not available in wgpu as of writing.

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

### Particle binning

Particles get scattered over time in memory (i.e. memory & spatial location diverge), resulting in slower and slower volume sampling & transfer over time. To counteract this, particles are binned to cells every n steps.
Strict sorting is not necessary, so it's done by counting all particles in each cell and then summing all cells up, creating prefix sums in each cell. Introducing some "sloppiness" in the spatial ordering of the cells allows to do this in a single pass (for details refer to the particle binning shader code).
### Solver

Using Preconditioned Conjugate Gradient solver for solving the poisson pressure equation (PPE). In comments and naming in the code I'm following the description in [Bridson's book](https://www.amazon.com/Simulation-Computer-Graphics-Robert-Bridson/dp/1568813260).
Implementing it in compute shader isn't entirely straight forward and needs some optimizing.
Blub is using an [Incomplete Poisson](https://software.intel.com/content/www/us/en/develop/articles/parallelized-incomplete-poisson-preconditioner-in-cloth-simulation.html) Preconditioner, better and shorter described by [Austin Eng here](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid#pressure-solve).

I started out with Jacobi iterations - very easy to implement, but inaccurate and slow (many iterations necessary). This is a good starting point though if you implement your own solver - [here's](https://github.com/Wumpf/blub/blob/c02ea18/shader/simulation/pressure_solve.comp) what the code looked like.

Looked into [A Multigrid Fluid Pressure SolverHandling Separating Solid Boundary Conditions, Chentanez et al. 2011](https://matthias-research.github.io/pages/publications/separatingBoundaries.pdf)
for a while but shied away from implementing such a complex solver at the moment without any reference code and with too little personal experience in the field.

#### Iteration Control

Typically solvers are run until a certain error threshold is reached.
Error in blub is expressed with an infinity norm (i.e. the max absolute residual; I used mean squared error previously, but max-error is more stable).
This is notoriously tricky on GPU, since this means that we need to feed back the error measure to determine how many more dispatch calls for solver iterations should be issued. We can't wait for the result as this would introduce a GPU-CPU stall. Experimenting with using error values from a couple of iterations ago (i.e. asynchronously querying the error) didn't yield promising results due to strong fluctuations and varying delay. Blub follows a different strategy instead:

There is a fix maximum number of iterations which determines how many compute dispatches are issued (note that there are several per iteration!), however most of these dispatches are indirect, so when evaluating the error, we may null out the indirect dispatch struct, making the remaining dispatches rather cheap (still not free though!).
Since evaluating the error itself is costly, this is done every couple of few iterations (configurable).

The last computed error and iteration count is queried asynchronously, in order to display a histogram in the gui and make informed choices for selecting the error tolerance, max iteration & error evaluation frequency parameters.

### Implicit Density Projection

For improved volume conversation & iteration times Blub implements a "secondary pressure solver" that uses fluid density instead of divergence as input. A video + paper can be found [here](https://animation.rwth-aachen.de/publication/0566/). I found that it improves the quality of the simulation tremendously for large timesteps (I typically run the simulation/solver at 120hz).

Compared to what is described (to my understanding) in the paper I made a few adjustments/trade-offs:
* For computing densities, neighboring solid cells are assumed to have a fixed (interpolation kernel derived) density contribution instead of sampling it with particles
* No resampling for degenerated cases
* No Push Boundaries (more details below)

To my knowledge this is the only publicly available implementation as of writing (I asked the authors for a look at their reference implementation but didn't get a reply.)

#### Push Boundaries

_Stopped using Push Boundaries in favor of dropping signed distance fields. Last use of Push Boundaries is at tag `last-use-of-push-boundaries`_

Push boundaries as described in the paper are a bit challenging for a GPU implementation as they require to store the maximum penetration into a solid cell, thus requiring in theory another vec3 volume with data that needs to be written atomically.
This displacement is clamped to the length of the cell (as in the paper) allowing us to accurately enough encode this value in a 32 bit value.
As particles only rarely exit the volume (this happens only due to inaccuracies in the first place), I employ a atomic compare exchange loop for writing the max displacement of a cell.

An interesting implementation detail of push boundaries is that we need to reserve a few more boundary cells than without: In the MAC grid one can usually keep outer border cells on the positive axis marked as free,
but with push boundaries we need to have a place to solid cell penetration for all borders.

### Boundary Handling

Fluid boundaries are handled using a realtime hull voxelization. Voxelization is done in a single pass using conservative rasterization.

This is rather unconventional, the more common approach is to use a signed distance field (earlier versions of Blub did exactly that, essentially everything at and before the `last-use-of-push-boundaries` tag).

*Disadvantage:* Boundary handling is rather coarse and it's very hard to properly particles that slip through.  
*Advantage:* Very fast for dynamic objects - getting signed distance field computation realtime for arbitrary meshes is very challenging.

## Rendering

Particle visualization with quads. Put a ridiculous amount of effort into to make the quads display perspective correct spheres.

The basic idea of screen space fluid rendering is very well described in these [GDC 2010 slides](http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf).
The implementation here is driven by a the depth filer described in [A Narrow-Range Filter for Screen-Space Fluid Rendering, Truong et al. 2018](http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf) which I tried to make reasonably efficient with some shared memory optimizations.  
On top of that comes some hand wavy (pun unintended) physically based rendering things, best check the comments in the shader code if you want to learn more ;-).

## Trivia

### Name
From German *[blubbern](https://en.wiktionary.org/wiki/blubbern)*, to bubble.  
Found out later that there was a [water park in Berlin](https://en.wikipedia.org/wiki/Blub_(water_park)) with that name, but it closed down in 2002.
