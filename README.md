# Blub

Experimenting with GPU driven 3D fluid simulation on the GPU using [WebGPU-rs](https://github.com/gfx-rs/wgpu-rs).  
Focusing primarily on hybrid approaches lagrangian/eularian approaches here (PIC/FLIP/APIC..).

For SPH (pure lagrangian) fluid simulation, check out my simple 2D DFSPH fluid simulator, [YASPH2D](https://github.com/Wumpf/yasph2d).

## Details (at this point random assortment thereof)

To learn more about fluid simulation in general, check out [my Gist on CFD](https://gist.github.com/Wumpf/b3e953984de8b0efdf2c65e827a1ccc3) where I gathered a lot of resources on the topic.

Implements currently APIC, [SIGGRAPH 2015, Jiang et al., The Affine Particle-In-Cell Method](https://www.math.ucla.edu/%7Ejteran/papers/JSSTS15.pdf).

### Particle to Grid Transfer

Transferring the particle's velocity to the grid is tricky/costly to do in parallel!
Either, velocities are scattered by doing 8 atomic adds for every particle to surrounding grid cells, or grid cells traverse all neighboring cells. (times 3 for staggered grid!)
There's some good ideas on how to do efficient scattering in [Ming et al 2018, GPU Optimization of Material Point Methods](http://www.cs.utah.edu/~kwu/GPU_MPM/GPU_MPM.pdf).

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

## Name

From German *[blubbern](https://en.wiktionary.org/wiki/blubbern)*, to bubble.  
Found out later that there was a [water park in Berlin](https://en.wikipedia.org/wiki/Blub_(water_park)) with that name, but it closed down in 2002.
