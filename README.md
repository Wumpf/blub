# Blub

Experimenting with GPU driven 3D fluid simulation on the GPU using [WebGPU-rs](https://github.com/gfx-rs/wgpu-rs).  
Focusing primarily on hybrid approaches lagrangian/eularian approaches here (PIC/FLIP/APIC..).

For SPH (pure lagrangian) fluid simulation, check out my simple 2D DFSPH fluid simulator, [YASPH2D](https://github.com/Wumpf/yasph2d).

## Application / Framework

### Build & Run

`cargo run`
Should work on Linux/Mac/Windows. (I'm developing on Windows, so things might break at random for the others)
Doing release mode (`cargo run --release`) gives quite a performance boost since I have shader optimizations turned off in non-optimized builds.

### Shaders

GLSL, compiled to SPIR-V at runtime. Shaders are hot reloaded on change, have fun!  
(on failure it will use the previously loaded shader)

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

Transferring the particle's velocity to the grid is tricky/costly to do in parallel!
Either, velocities are scattered by doing 8 atomic adds for every particle to surrounding grid cells, or grid cells traverse all neighboring cells. (times 3 for staggered grid!)
There's some good ideas on how to do efficient scattering in [Ming et al 2018, GPU Optimization of Material Point Methods](http://www.cs.utah.edu/~kwu/GPU_MPM/GPU_MPM.pdf).
Note though that today atomic floats addition is pretty much only available in CUDA and OpenGL (using an NV extension)!

In Blub I tried something new (to my knowledge):
Particles form a linked list by putting their index with a atomic exchange operation in a "linked list head pointer grid" which is a grid dual to the main velocity volume.
Then, every velocity grid cell walks 8 neighboring linked lists to gather velocity.

Note that this all makes MAC/staggered grids a lot less appealing since the volume in which particles need to be accumulated gets bigger & more complicated, i.e. a lot slower.
After various tries with collocated grids I ended up using staggered after all (for some details see #14) since I couldn't get the collocated case quite right.
(to avoid artifacts with collocated grids, Rhie-Chow interpolation is required. It's widespread in CFD since collocated grids are required for arbitrary meshes, but it's hard to find any resources in the computer graphics community [...])  
Generally, one can either stick with a single linked list grid (sampling 6 different but overlapping cells per velocity component) or three different linked-list grids.
I eventually settled with three different grids, processing a single velocity component at a time.

Note that _by far_ the biggest bottleneck in this approach is walking the particle linked list. Doing a shared memory optimization yielded >4x performance speed up:
Every thread walks only a single linked list, stores the result to shared memory and then reads the remaining seven neighbor linked lists from shared memory. 👌

### Velocity Extrapolation

Typical implementations of PIC/FLIP/APIC include a velocity extrapolation step which extends velocities from fluid cells into air and (with some tweaks) solid cells.
This is done in order to...
* fix discrete [divergence](https://en.wikipedia.org/wiki/Divergence)
    * think of a falling droplet, modeled as a single fluid cell with downward velocity. As there's not other forces, our tiny fluid is divergence free. If we were to take central differences of velocity with the surrounding cells as is though we would come to a different conclusion!
* particle advection
  * particles interpolating velocities, thus grabbing velocity from solid/fluid cells.
  * particles leaving fluid cells during advection
    * advection is usually done via higher order differential equation solver which may sample the velocity grid outside of the cell any particular particle started in
* useful for some kind of renderings (I believe)

Extrapolation in Blub:
* divergence computation
  * fix on the fly by looking into marker grid (doesn't go far, so this is rather cheap)
* particle advection
  * do a single pass extrapolation
    * it's a bit more complex than normal extrapolation schemes since in order to get around double buffering velocities/markers we do everything in a single pass.
* don't use anything fancy that needs velocity elsewhere 🙂

### Solver

Using Preconditioned Conjugate Gradient solver for solving the poisson pressure equation (PPE). In comments and naming in the code I'm following the description in [Bridson's book](https://www.amazon.com/Simulation-Computer-Graphics-Robert-Bridson/dp/1568813260).
Implementing it in compute shader isn't entirely straight forward and needs some optimizing.
Blub is using an [Incomplete Poisson](https://software.intel.com/content/www/us/en/develop/articles/parallelized-incomplete-poisson-preconditioner-in-cloth-simulation.html) Preconditioner, better and shorter described by [Austin Eng here](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid#pressure-solve).

I started out with Jacobi iterations - very easy to implement, but inaccurate and slow (many iterations necessary). This is a good starting point though if you implement your own solver - [here's](https://github.com/Wumpf/blub/blob/c02ea18/shader/simulation/pressure_solve.comp) what the code looked like.

Looked into [A Multigrid Fluid Pressure SolverHandling Separating Solid Boundary Conditions, Chentanez et al. 2011](https://matthias-research.github.io/pages/publications/separatingBoundaries.pdf)
for a while but shied away from implementing such a complex solver at the moment without any reference code and with too little personal experience in the field.


## Rendering

Particle visualization with quads. Put a ridiculous amount of effort into to make the quads display perspective correct spheres.

The basic idea of screen space fluid rendering is very well described in these [GDC 2010 slides](http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf).
The implementation here is driven by a the depth filer described in [A Narrow-Range Filter for Screen-Space Fluid Rendering, Truong et al. 2018](http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf) which I tried to make reasonably efficient with some shared memory optimizations.  
On top of that comes some hand wavy (pun unintended) physically based rendering things, best check the comments in the shader code if you want to learn more ;-).

## Trivia

### Name
From German *[blubbern](https://en.wiktionary.org/wiki/blubbern)*, to bubble.  
Found out later that there was a [water park in Berlin](https://en.wikipedia.org/wiki/Blub_(water_park)) with that name, but it closed down in 2002.
