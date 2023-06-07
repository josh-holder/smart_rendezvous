import jax.numpy as jnp

# Define thruster positions
thruster_positions = jnp.array(([0.5,0,0],
                                [0.5,0,0],
                                [-0.5,0,0],
                                [-0.5,0,0],
                                [0,0.5,0],
                                [0,0.5,0],
                                [0,-0.5,0],
                                [0,-0.5,0],
                                [0,0,0.5],
                                [0,0,0.5],
                                [0,0,-0.5],
                                [0,0,-0.5]))

#Define thruster force vectors, with them pointing 45 degrees from the surface normal.
thruster_force_vectors = -1*jnp.array(([0.707, 0.707, 0],
                                    [0.707, -0.707, 0],
                                    [-0.707, 0.707, 0],
                                    [-0.707, -0.707, 0],
                                    [0, 0.707, 0.707],
                                    [0, 0.707, -0.707],
                                    [0, -0.707, 0.707],
                                    [0, -0.707, -0.707],
                                    [0.707, 0, 0.707],
                                    [-0.707, 0, 0.707],
                                    [0.707, 0, -0.707],
                                    [-0.707, 0, -0.707]))

u, d, v = svd(B)
