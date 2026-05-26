import os
import numpy as np
import matplotlib.pyplot as plt

from dlroms import *

try:
    import dolfin as fe
    DOLFIN_AVAILALBE = True
except ImportError:
    fe = None
    DOLFIN_AVAILALBE = False
    print("DOLFIN not available.")


def stokes(mesh, inflows, outflows, source=[0.0, 0.0]):
    from fenics import FiniteElement, NodalEnrichedElement, FunctionSpace, VectorElement, TrialFunctions, TestFunctions
    from fenics import inner, grad, dx, div, assemble, DirichletBC, Constant, Expression
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import csr_matrix

    pP1  = FiniteElement("CG", mesh.ufl_cell(), 1)
    vP1B = VectorElement(NodalEnrichedElement(FiniteElement("CG", mesh.ufl_cell(), 1),
                                              FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)))

    pspace, vspace = pP1, vP1B
    W = FunctionSpace(mesh, vspace * pspace)
    W0 = FunctionSpace(mesh, vspace)
    (b, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    space = fe.space(mesh, "CG", 1, vector_valued=True, bubble=True)
    f = fe.interpolate(source, space)

    a = inner(grad(b), grad(v))*dx - div(v)*p*dx - q*div(b)*dx
    L = inner(f, v)*dx

    def outflow(x):
        result = False
        for out in outflows:
            result = result or out(x)
        return result

    def inflow(x):
        result = False
        for infl in inflows:
            result = result or infl[0](x)
        return result

    def walls(x):
        return not(outflow(x) or inflow(x))

    noslip = DirichletBC(W.sub(0), Constant((0.0, 0.0)), lambda x, on: on and walls(x))
    def make_bc(i):
        return DirichletBC(W.sub(0), fe.interpolate(inflows[i][1], W0), lambda x, on: on and inflows[i][0](x))
    ins = [make_bc(i) for i in range(len(inflows))]

    A = assemble(a)
    F = assemble(L)

    for bc in [noslip, *ins]:
        bc.apply(A)
        bc.apply(F)

    A = csr_matrix(A.array())
    F = F[:]

    bp = spsolve(A, F)
    bp_f = fe.asfunction(bp, W)

    from fenics import dof_to_vertex_map
    Vh = fe.space(mesh,'CG', 1)
    Vb = fe.space(mesh,'CG', 1, vector_valued=True)
    nvertices = mesh.coordinates().shape[0]
    b = bp_f.compute_vertex_values(mesh)[:2*nvertices].reshape(2, -1).T
    indexes = dof_to_vertex_map(Vh)
    b = b[indexes].reshape(-1)
    return b

domain = fe.rectangle((0, 0), (3, 2)) - fe.circle((0.95, 0.7), 0.25) - fe.circle((1.5, 1.35), 0.25) - fe.circle((2.05, 0.7), 0.25)
mesh = fe.mesh(domain, stepsize=0.05)

left = lambda x: x[0] < 1e-6
right = lambda x: 3 - x[0] < 1e-12
circle = lambda x0, r: (lambda x: ((x[0]-x0[0])**2 + (x[1]-x0[1])**2)**0.5 < r + 1e-3)

def inflows(b0x, b0y, b1, b2, b3):
    return [
        [left, lambda x: [b0x*np.exp(-25*(x[1]-1)**2), b0y*np.exp(-25*(x[1]-1)**2)]],
        [circle((0.95, 0.70), 0.25), lambda x: [b1*(-x[1]+0.70), b1*(x[0]-0.95)]],
        [circle((1.50, 1.35), 0.25), lambda x: [b2*(-x[1]+1.35), b2*(x[0]-1.50)]],
        [circle((2.05, 0.70), 0.25), lambda x: [b3*(-x[1]+0.70), b3*(x[0]-2.05)]]
    ]

outflows = [right]

def bi(i):
    e = np.zeros(5)
    e[i] = 1
    return stokes(mesh, inflows(*e), outflows)

# Calcolo immediato all'importazione
b0x, b0y, b1, b2, b3 = [bi(i) for i in range(5)]
Vh = fe.space(mesh, 'CG', 1)
Vb = fe.space(mesh, 'CG', 1, vector_valued=True)
clc()

def ADR(eps, theta, c1, c2, c3):
    from fenics import grad, inner, dx, solve, TrialFunction, TestFunction, Function, Constant, DirichletBC, assemble
    b = fe.asfunction(b0x*np.cos(theta) + b0y*np.sin(theta) + c1*b1 + c2*b2 + c3*b3, Vb)
    u, v = TrialFunction(Vh), TestFunction(Vh)
    L = eps*inner(grad(u), grad(v))*dx + inner(b, grad(u))*v*dx
    f = Constant(0.0)*v*dx
    bc = DirichletBC(Vh, fe.interpolate(lambda x: np.exp(-25*(x[1]-1)**2), Vh), left)
    u = Function(Vh)
    solve(L == f, u, bc)
    clc()
    return u.vector()[:]

if __name__ == "__main__":
    import torch
    from joblib import Parallel, delayed
    from tqdm import tqdm

    N_SAMPLES = 6400
    
    eps_vals = np.random.uniform(0.01, 0.1, N_SAMPLES)
    theta_vals = np.random.uniform(0, 2*np.pi, N_SAMPLES)
    c1_vals = np.random.uniform(-1, 1, N_SAMPLES)
    c2_vals = np.random.uniform(-1, 1, N_SAMPLES)
    c3_vals = np.random.uniform(-1, 1, N_SAMPLES)

    def compute_sample(i):
        return ADR(eps_vals[i], theta_vals[i], c1_vals[i], c2_vals[i], c3_vals[i])

    u_results = Parallel(n_jobs=2)(delayed(compute_sample)(i) for i in tqdm(range(N_SAMPLES)))

    u_array = np.array(u_results, dtype=np.float32)

    eps_t = torch.tensor(eps_vals, dtype=torch.float32).unsqueeze(1)
    theta_t = torch.tensor(theta_vals, dtype=torch.float32).unsqueeze(1)
    c1_t = torch.tensor(c1_vals, dtype=torch.float32).unsqueeze(1)
    c2_t = torch.tensor(c2_vals, dtype=torch.float32).unsqueeze(1)
    c3_t = torch.tensor(c3_vals, dtype=torch.float32).unsqueeze(1)

    mu_array = torch.cat((eps_t, theta_t, c1_t, c2_t, c3_t), dim=1).numpy()

    # os.makedirs('data/raw', exist_ok=True)
    
    # np.save('data/raw/u_data.npy', u_array)
    # np.save('data/raw/mu_data.npy', mu_array)