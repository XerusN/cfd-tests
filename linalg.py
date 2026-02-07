import numpy as np

def jacobi(a, b, x):
    
    np.set_printoptions(formatter={'float': lambda x: "{0:2.0f}".format(x)})

    print(a)
    print(b)
    
    n = len(x)
    
    x_t = np.zeros(n)
    x_t[:] = x[:]
    r = np.zeros(n)
    
    norm = 1.
    it = 0
    norm_0 = np.linalg.norm(b)
    
    while norm > 1e-4:
        
        for i in range(n):
            s1 = np.dot(a[i, :i], x_t[:i])
            s2 = np.dot(a[i, i + 1:], x_t[i + 1:])
            x[i] = (b[i] - s1 - s2) / a[i, i]
        
        for i in range(n):
            r[i] = b[i] - np.sum(a[i, :]*x[:])
        
        x_t[:] = x[:]
        norm = np.linalg.norm(r)/norm_0
        
        it+= 1
        print(it, norm)
        
        if it > 100:
            raise ValueError('Did not converge')
    
    return x