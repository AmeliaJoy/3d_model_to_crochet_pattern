import gpytoolbox as gpy, numpy as np, polyscope as ps,scipy as sp

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def vertex_area(V,F):
    face_area = gpy.doublearea(V,F) / 6 #1/3 of the area of the triangle
    k = np.zeros(len(V))
    face_area = face_area.reshape(face_area.shape[0],1)
    face_matrix = np.concatenate((face_area,face_area,face_area), axis=1)   
    np.add.at(k,F,face_matrix)
    i = np.arange(0,len(V))
    A = sp.sparse.csr_matrix((k, (i,i)))
    return A
def inv_vertex_area(V,F):
    face_area = gpy.doublearea(V,F) / 6 #1/3 of the area of the triangle
    k = np.zeros(len(V))
    face_area = face_area.reshape(face_area.shape[0],1)
    face_matrix = np.concatenate((face_area,face_area,face_area), axis=1)   
    np.add.at(k,F,face_matrix)
    k = 1/k
    i = np.arange(0,len(V))
    A = sp.sparse.csr_matrix((k, (i,i)))
    return A
def edge_counterclockwise(P,V, E):
    vec = V[E[:,:,1]]-V[E[:,:,0]]
    vec2 = V[E[:,:,0]]-V[P]
    normals = gpy.per_face_normals(V,P)
    signs = (np.sign(np.sum(np.divide(np.cross(vec2,vec),normals[:,None],where=normals[:,None]!=0),axis=2)[:,0]))
    vectors = vec*signs[:,None,None]
    return vec
def ugrad(V,F,U):
    Uface = U[F,None] #this makes an array with the same shape as the faces: that is, 2-dimensional with the first dimension the triangle and the second dimension the indices: and then basically creates array with U(i) in all these spots. Then, none gives it a third dimension so I can multiply it by all of the components of the edge vector.
    N = gpy.per_face_normals(V,F)[:,None]
    halfedges = gpy.halfedges(F) 
    face_area = (gpy.doublearea(V,F)/2)[:,None]
    edges = edge_counterclockwise(F,V,halfedges)

    
    gradu = (0.5/face_area)
    faceperi = (Uface*np.cross(N,edges)) 
    gradu = gradu*(np.sum(faceperi,axis=1))#this sums up everything along the second axis. the first axis would be summing up all the triangles into one, the second sums all the vectors into one, and the third sums all the coordinates of each vector into one.
    return gradu
def xvec(gradu):
    magnitudes = np.linalg.norm(gradu, axis=1)[:,None]

    return(-gradu/magnitudes)
def directed_halfedges(F):
    halfedge_list = np.stack((
        np.stack((np.stack((F[:,0],F[:,1]),axis=1),np.stack((F[:,0],F[:,2]),axis=1)),axis=1),
        np.stack((np.stack((F[:,1],F[:,0]),axis=1),np.stack((F[:,1],F[:,2]),axis=1)),axis=1),
        np.stack((np.stack((F[:,2],F[:,0]),axis=1),np.stack((F[:,2],F[:,1]),axis=1)),axis=1)
        ),axis=1)
    return halfedge_list
def divxlengths(V,F):
    halfedge_list = directed_halfedges(F)
    halfedge_dist = V[halfedge_list[:,:,:,1]]-V[halfedge_list[:,:,:,0]]
    return(halfedge_dist)
def divx(V,F,xvec):

    e = divxlengths(V,F)
    cot = np.reciprocal(np.tan(gpy.tip_angles(V, F)))
    cot_list = np.stack((
        np.stack((cot[:,2],cot[:,1]),axis=1),
        np.stack((cot[:,2],cot[:,0]),axis=1),
        np.stack((cot[:,1],cot[:,0]),axis=1)
        ),axis=1)
    dp = (np.sum(e*xvec[:,None,None,:],axis=3))

    divx = 0.5*(np.sum((cot_list*dp),axis=2)) 

    vertices = np.zeros(V.shape[0])
    (np.add.at(vertices,F,divx))
    return vertices
    
V,F = gpy.read_mesh("data/pixelbun.obj")

#u = np.ones(V.shape[0]) #gives array of ones with length of the shape of the vertices,  the shape of the vertices is jsut the number of vertices
U0 = np.zeros(V.shape[0])
U0[154] = 10000000000000000

Lc = -gpy.cotangent_laplacian(V, F)
A = (vertex_area(V,F))

invA = (inv_vertex_area(V,F))
t = 0.00000002
Lc = invA*Lc
H = sp.sparse.linalg.spsolve(A - t*Lc, U0)

gradu = (ugrad(V,F,H))

xvec = xvec(gradu)
divx = divx(V,F,xvec)
Lc = -gpy.cotangent_laplacian(V, F)
geodesic = sp.sparse.linalg.spsolve(Lc, divx)

print(geodesic)
ps.init()
ps_penguin = ps.register_surface_mesh("bunny", V, F,
    material='wax')
ps_penguin.add_scalar_quantity("per vertex y coord", geodesic,enabled=True,isolines_enabled=True)
ps_penguin.add_vector_quantity("vecs ambient", xvec/100, vectortype='ambient',defined_on="faces")
ps.show()

