import polyscope as ps, gpytoolbox as gpy
V,F = gpy.icosphere(2)
ps.init()
ps.register_surface_mesh("sphere", V, F)
ps.show()
#ps.init() to start