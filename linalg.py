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
    
    while norm > 1e-4:
        
        for i in range(n):
            x[i] = b[i]
            for j in range(n):
                if i == j:
                    continue
                
                x[i] -= a[i,j]*x_t[i]
            
            x[i] /= a[i, i]
        
        for i in range(n):
            r[i] = b[i] - np.sum(a[i, :]*x[:])
        
        x_t[:] = x[:]
        norm = np.linalg.norm(r)
        
        it+= 1
        print(it, norm)
        
        if it > 100:
            raise ValueError('Did not converge')
    
    return x