#ifndef LLGRID_WRITEACCESS
#define LLGRID_ACCESS restrict readonly
#else
#define LLGRID_ACCESS restrict LLGRID_WRITEACCESS
#endif

// Dual grid linked list
// Our dual grid has a positive 0.5 offset. I.e. its origin relative to the main grid is at (0.5, 0.5, 0.5)
// This also means that the main grid cell at (0,0,0) has only one neighbor in dual, whereas the cell at (gridsize-1) has all 8
layout(set = 2, binding = 0, r32ui) uniform LLGRID_ACCESS uimage3D LinkedListDualGrid;

// Fluid volumes (both variables point to different ones!)
// Origin texel represents position (0,0,0), NOT as one might expect (-0.5 / textureSize)
layout(set = 3, binding = 0, rgba32f) uniform restrict image3D VelocityGridWrite;
layout(set = 3, binding = 1) uniform texture3D VelocityGridRead;

// Occupancy calculator: https://xmartlabs.github.io/cuda-calculator/
#define COMPUTE_PASS_VOLUME layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;