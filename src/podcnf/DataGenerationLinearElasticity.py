import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import dolfin
    from fenics import *
    from ufl_legacy import nabla_div
    from dlroms import *
    from dlroms import fe
    from dlroms.gp import GaussianRandomField
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import csr_matrix
    from IPython.display import clear_output as clc
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False

def check_fenics():
    if not FENICS_AVAILABLE:
        raise RuntimeError(
            "FEniCS non è installato in questo ambiente. "
            "DataGenerationElastic.py richiede FEniCS. "
            "Esegui questo script su Google Colab o in un container Docker dedicato."
        )

if FENICS_AVAILABLE:
    mesh = fe.unitsquaremesh(30, 30) # square mesh of 30x30 unit
    Vh = fe.space(mesh, 'CG', 1, vector_valued = True) # Functional space with Continuous Galerkin
    V_lam = fe.space(mesh, 'DG', 0) # 0 means constant polynomials -> constant property in each triangle
    V_aux = fe.space(mesh, 'CG', 1)

    G = GaussianRandomField(mesh, kernel = lambda r: np.exp(-25*r**2), upto = 50)
    # G = GaussianRandomField(mesh, kernel = lambda r: np.exp(-25*r**2), upto = 15)
    # G = GaussianRandomField(mesh, kernel = lambda r: np.exp(-5*r**2), upto = 15)

    parameters = {'Parameter':['rho', 'lambda', 'nu', 'm', 'delta', 'x0', 'theta'],
                'Min': [0.5, 0.9, 0.5, 1.0, 0.05, 0.3, np.pi/4.0],
                'Max': [2.0, 1.1, 1.0, 1.5, 0.10, 0.7, 3*np.pi/4.0],
                'Meaning': ['Body force (shield)', 'First Lamé parameter (shield)',
                            'Second Lamé parameter (shield)', 'Mass (hail grain)', 'Diameter (hail grain)',
                            'Impact location', 'Angle of impact']}

def FOMsampler(stochastic_seed, mass, delta, option:int = 0):
    check_fenics()
    
    rho = 1.0
    nu = 1.0
    x0 = 0.5
    theta = np.pi/2
    material_angle = None 

    if option == 0:
        g = (G.sample(stochastic_seed) > 0)*7 + 0.1
        lambda_ = interpolate(fe.asfunction(g, V_aux), V_lam)
        lambda_ = fe.asfunction((lambda_.vector()[:] < 4)*7 + 0.1, V_lam)

    elif option == 1:
        rng = np.random.RandomState(stochastic_seed)
        m = rng.uniform(0, 2 * np.pi)
        material_angle = m

        x_c, y_c = 0.5, 0.5
        cos_m = np.cos(m)
        sin_m = np.sin(m)

        l1 = 7.1
        l2 = 0.1

        linear_expression = lambda x: np.where(
            (x[1] - y_c) * cos_m - (x[0] - x_c) * sin_m > 0,
            l1,
            l2
        )
        lambda_ = fe.interpolate(linear_expression, V_lam)

    else:
        raise ValueError("No Valid Option!")

    # Boundary conditions
    tol = 1e-14
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[1]<tol
    bc = DirichletBC(Vh, Constant((0, 0)), clamped_boundary)

    # Auxiliary definitions
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    def sigma(u):
        return lambda_*nabla_div(u)*Identity(2) + 2*nu*epsilon(u)

    # Variational problem
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    f = Constant((0, -rho))
    T = fe.interpolate(lambda x: [mass*np.cos(theta)*(x[1] > 0.99)*(np.abs(x[0]-x0)<delta),
                                  -mass*np.sin(theta)*(x[1] > 0.99)*(np.abs(x[0]-x0)<delta)], Vh)
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds

    # Assembling and adjusting
    A = assemble(a)
    F = assemble(L)
    bc.apply(A)
    bc.apply(F)
    A = csr_matrix(A.array())
    F = F[:]

    # Solving
    u_sol = spsolve(A, F)
    clc()
    return mass, delta, lambda_.vector()[:], u_sol, material_angle