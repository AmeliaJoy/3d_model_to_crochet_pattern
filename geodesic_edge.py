import potpourri3d as pp3d #implement myself later for practice
import numpy as np
import polyscope as ps
import gpytoolbox as gpy
import geodesic_distance as gd
def rodrigues_rotation(v,k,theta):
    v_rot = v*np.cos(theta) + np.cross(k,v)* np.sin(theta) + ((np.einsum('ij, ij->i', k, v))*(1-np.cos(theta)))[:,np.newaxis]*k
    return v_rot
seed=83
V,F = pp3d.read_mesh("data/penguin.obj")
distance_solver = pp3d.MeshHeatMethodDistanceSolver(V,F)
path_solver = pp3d.EdgeFlipGeodesicSolver(V,F)

dist = distance_solver.compute_distance(seed)
max = np.argmax(dist)
path = (path_solver.find_geodesic_path(v_start=seed, v_end=max))


gradu = (gd.ugrad(V,F,dist))
distance_gradient = np.zeros(V.shape)
gradu = (np.repeat(gradu,3,axis=0).reshape(F.shape[0],3,3))
np.add.at(distance_gradient,F,gradu)
#V; 3512 F: 17551,24271,4111,751,20911,7471

#rodrigues rotation formula
normals = gpy.per_vertex_normals(V,F)
v_rots = gd.xvec(rodrigues_rotation(distance_gradient,normals,-np.pi/2))



ps.init()
ps_penguin = ps.register_surface_mesh("penguin", V, F,
    material='wax')
ps_penguin.add_scalar_quantity("geodesic distance from seed", dist,enabled=True,isolines_enabled=True)

ps_penguin.add_vector_quantity("gradient vecs (gradu)",distance_gradient, enabled=True, defined_on="vertices")
ps_penguin.add_vector_quantity("rotated gradient",v_rots, enabled=True, defined_on="vertices")
ps.show()

