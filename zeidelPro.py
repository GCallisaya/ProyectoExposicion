import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()
    
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new
    
    return x

# Datos del circuito eléctrico
A = np.array([
    [9, -3, -4],
    [-3, 8, 0],
    [-4, -5, 9]
])
b = np.array([10, 20, 0])

# Vector inicial (puede ser un vector de ceros o una aproximación inicial)
x0 = np.zeros(len(b))

# Resolución del sistema usando el método de Gauss-Seidel
x = gauss_seidel(A, b, x0)

print("Corrientes (I1, I2, I3):", x)

