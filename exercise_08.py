import gpytoolbox as gpy, numpy as np, polyscope as ps,scipy as sp
V,F = gpy.read_mesh("data/penguin.obj")
L = gpy.cotangent_laplacian(V, F)
u = np.ones(V.shape[0]) #gives array of ones with length of the shape of the vertices,  the shape of the vertices is jsut the number of vertices
print(f"The Laplacian of a constant function: {np.dot(u, L*u)}")

V0,F = gpy.read_mesh("data/noisy_penguin.obj")
L = gpy.cotangent_laplacian(V0, F)
M = gpy.massmatrix(V0, F)
t = 5e-3
V = sp.sparse.linalg.spsolve(M + t*L, M*V0)
print(M*V0)

print(type(M))
print(M.shape)
print(type(V0))
print(V0.shape)
ps.init()
ps_penguin = ps.register_surface_mesh("denoised penguin", V, F,
    material='wax')
ps.show()