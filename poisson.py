import numpy as np

GRAD_ITER_MAX = 2

def poisson(mesh, boundaries):
    # Only Dirichlet bnd for now
    
    
    cells = mesh['cells']
    volumes = cells['volumes']
    pairs = mesh['pairs']
    n_cells = cells['n']
    n_pairs = pairs['n']
    
    phi = np.zeros(n_cells)
    phi_edges = np.zeros(n_pairs)
    grad_x = np.zeros(n_cells)
    grad_y = np.zeros(n_cells)
    
    matrix = np.zeros((n_cells, n_cells))
    rhs = np.zeros(n_cells)
    
    
    # Init
    for i in range(n_cells):
        phi[i] = cells['centers'][i][0] + cells['centers'][i][1]

    def compute_grad():
        
        for _ in range(GRAD_ITER_MAX):
            # Face values
            for i in range(n_pairs):
                if pairs['on_bnd'][i]:
                    continue
                cell_1 = pairs['neighboring_cells'][i][0]['Cell']
                cell_2 = pairs['neighboring_cells'][i][1]['Cell']
        
                phi_edges[i] = (phi[cell_1] + phi[cell_2])*0.5
                middle_x = (cells['centers'][cell_1][0] + cells['centers'][cell_2][0])/2.
                middle_y = (cells['centers'][cell_1][1] + cells['centers'][cell_2][1])/2.
                phi_edges[i] += 0.5*((grad_x[cell_1] + grad_x[cell_2])*(pairs['centers'][i][0] - middle_x) + (grad_y[cell_1] + grad_y[cell_2])*(pairs['centers'][i][1] - middle_y))
                
                # TODO: bnd handling
                
            # Grads
            grad_x[:] = 0.
            grad_y[:] = 0.
            
            for i in range(n_pairs):
                # TODO: not correct bnd handling
                if pairs['on_bnd'][i]:
                    continue
                
                cell_1 = pairs['neighboring_cells'][i][0]['Cell']
                cell_2 = pairs['neighboring_cells'][i][1]['Cell']
                
                area = pairs['cells_areas'][i]
                flux_x = area * phi_edges[i] * pairs['cells_normals'][i][0]
                flux_y = area * phi_edges[i] * pairs['cells_normals'][i][1]
                
                grad_x[cell_1] += flux_x/volumes[cell_1]
                grad_y[cell_1] += flux_y/volumes[cell_1]
                
                grad_x[cell_2] -= flux_x/volumes[cell_2]
                grad_y[cell_2] -= flux_y/volumes[cell_2]
        
        return
    
    compute_grad()
    
    return phi, phi_edges, grad_x, grad_y