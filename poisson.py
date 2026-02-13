import numpy as np
from linalg import jacobi

GRAD_ITER_MAX = 2

def default_init(x, y):
    return x + y

def poisson_cell(mesh, boundaries, init=default_init):
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
    grad_x_e = np.zeros(n_pairs)
    grad_y_e = np.zeros(n_pairs)
    
    matrix = np.zeros((n_cells, n_cells))
    rhs = np.zeros(n_cells)
    
    # Init
    for i in range(n_cells):
        phi[i] = init(cells['centers'][i][0], cells['centers'][i][1])

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
                
            for i_bnd, bnd in enumerate(boundaries):
                if bnd[0] == 'D':
                    for edge in mesh['boundaries']['faces'][i_bnd]:
                        # print(edge)
                        # print(bnd[1])
                        phi_edges[edge] = bnd[1]
                else:
                    raise ValueError("Bnd not implemented")
                
            # Grads
            grad_x[:] = 0.
            grad_y[:] = 0.
            
            for i in range(n_pairs):
                area = pairs['cells_areas'][i]
                flux_x = area * phi_edges[i] * pairs['cells_normals'][i][0]
                flux_y = area * phi_edges[i] * pairs['cells_normals'][i][1]
                if pairs['on_bnd'][i]:
                    if 'Cell' in pairs['neighboring_cells'][i][0]:
                        cell_1 = pairs['neighboring_cells'][i][0]['Cell']
                        grad_x[cell_1] += flux_x/volumes[cell_1]
                        grad_y[cell_1] += flux_y/volumes[cell_1]
                    else:
                        cell_2 = pairs['neighboring_cells'][i][1]['Cell']
                        grad_x[cell_2] -= flux_x/volumes[cell_2]
                        grad_y[cell_2] -= flux_y/volumes[cell_2]
                else:
                    cell_1 = pairs['neighboring_cells'][i][0]['Cell']
                    cell_2 = pairs['neighboring_cells'][i][1]['Cell']
                
                    grad_x[cell_1] += flux_x/volumes[cell_1]
                    grad_y[cell_1] += flux_y/volumes[cell_1]
                    
                    grad_x[cell_2] -= flux_x/volumes[cell_2]
                    grad_y[cell_2] -= flux_y/volumes[cell_2]
                    
        for i in range(n_pairs):
            if pairs['on_bnd'][i]:
                continue
            cell_1 = pairs['neighboring_cells'][i][0]['Cell']
            cell_2 = pairs['neighboring_cells'][i][1]['Cell']
            
            d_cf_x = cells['centers'][cell_2][0] - cells['centers'][cell_1][0]
            d_cf_y = cells['centers'][cell_2][1] - cells['centers'][cell_1][1]
            d_cf_norm = np.sqrt(d_cf_x**2 + d_cf_y**2)
            e_cf_x = d_cf_x/d_cf_norm
            e_cf_y = d_cf_y/d_cf_norm
    
            gw = np.sqrt((pairs['centers'][i][0] - cells['centers'][cell_2][0])**2 + (pairs['centers'][i][1] - cells['centers'][cell_2][1])**2)/d_cf_norm
            
            grad_x_e[i] = grad_x[cell_1]*gw + grad_x[cell_2]*(1.-gw)
            grad_y_e[i] = grad_y[cell_1]*gw + grad_y[cell_2]*(1.-gw)
            
            grad_x_e[i] = grad_x_e[i] + e_cf_x*((phi[cell_2] - phi[cell_1])/d_cf_norm - (grad_x_e[i]*e_cf_x + grad_y_e[i]*e_cf_y))
            grad_y_e[i] = grad_y_e[i] + e_cf_y*((phi[cell_2] - phi[cell_1])/d_cf_norm - (grad_x_e[i]*e_cf_x + grad_y_e[i]*e_cf_y))
            
        for i_bnd, bnd in enumerate(boundaries):
            if bnd[0] == 'D':
                for i_edge, edge in enumerate(mesh['boundaries']['faces'][i_bnd]):
                    cell = mesh['boundaries']['cells'][i_bnd][i_edge]
                    d_x = pairs['centers'][edge][0] - cells['centers'][cell][0]
                    d_y = pairs['centers'][edge][1] - cells['centers'][cell][1]
                    if 'Cell' in pairs['neighboring_cells'][edge][0]:
                        n_x = pairs['cells_normals'][edge][0]
                        n_y = pairs['cells_normals'][edge][1]
                    else:
                        n_x = - pairs['cells_normals'][edge][0]
                        n_y = - pairs['cells_normals'][edge][1]
                        
                    # TODO: make this better (take into account cross gradients)
                    grad_x_e[edge] = (bnd[1] - phi[cell])/(-d_x*n_x - d_y*n_y) * n_x
                    grad_y_e[edge] = (bnd[1] - phi[cell])/(-d_x*n_x - d_y*n_y) * n_y
            else:
                raise ValueError("Bnd not implemented")
            
        
        
        return
    
    def laplacian():
        matrix[:, :] = 0.
        rhs[:] = 0.
        
        for i in range(n_pairs):
            if pairs['on_bnd'][i]:
                continue
            cell_1 = pairs['neighboring_cells'][i][0]['Cell']
            cell_2 = pairs['neighboring_cells'][i][1]['Cell']
            
            s = pairs['cells_areas'][i]
            
            e_f_x = cells['centers'][cell_2][0] - cells['centers'][cell_1][0]
            e_f_y = cells['centers'][cell_2][1] - cells['centers'][cell_1][1]
            d_cf_norm = np.sqrt(e_f_x**2 + e_f_y**2)
            e_f_x *= s/d_cf_norm
            e_f_y *= s/d_cf_norm
            s_f_x = pairs['cells_normals'][i][0]*s
            s_f_y = pairs['cells_normals'][i][1]*s
            t_f_x = s_f_x - e_f_x
            t_f_y = s_f_y - e_f_y
            
            flux_f = - s / d_cf_norm
            
            matrix[cell_1, cell_1] -= flux_f
            matrix[cell_1, cell_2] += flux_f
            
            matrix[cell_2, cell_2] -= flux_f
            matrix[cell_2, cell_1] += flux_f
            
            rhs[cell_1] -= grad_x_e[i] * t_f_x + grad_y_e[i] * t_f_y
            rhs[cell_2] += grad_x_e[i] * t_f_x + grad_y_e[i] * t_f_y
            
        for i_bnd, bnd in enumerate(boundaries):
            if bnd[0] == 'D':
                for edge, cell in zip(mesh['boundaries']['faces'][i_bnd], mesh['boundaries']['cells'][i_bnd]):
                    s = pairs['cells_areas'][edge]
                    
                    if 'Cell' in pairs['neighboring_cells'][edge][0]:
                        sign = 1.
                    else:
                        sign = -1.
                        
                    n_x = pairs['cells_normals'][edge][0]*sign
                    n_y = pairs['cells_normals'][edge][1]*sign
                    s_x = s*n_x
                    s_y = s*n_y
                    
                    e_f_x = (pairs['centers'][edge][0] - cells['centers'][cell][0])
                    e_f_y = (pairs['centers'][edge][1] - cells['centers'][cell][1])
                    d_cf_norm = np.sqrt(e_f_x**2 + e_f_y**2)
                    e_f_x *= s/d_cf_norm
                    e_f_y *= s/d_cf_norm
                    s_f_x = n_x*s
                    s_f_y = n_y*s
                    t_f_x = s_f_x - e_f_x
                    t_f_y = s_f_y - e_f_y
                    
                    flux_b = s / d_cf_norm
                    
                    matrix[cell, cell] += flux_b
                    
                    rhs[cell] += flux_b*bnd[1] + (grad_x_e[i] * t_f_x + grad_y_e[i] * t_f_y)
            else:
                raise ValueError("Bnd not implemented")
        
    # for i in range(4):
    #     compute_grad()
        
    #     laplacian()
        
    #     phi = np.linalg.solve(matrix, rhs)
    
    compute_grad()
        
    laplacian()
    phi = jacobi(matrix, rhs, phi)
    
    compute_grad()
    
    laplacian()
    phi = jacobi(matrix, rhs, phi)
    
    compute_grad()
    
    return phi, phi_edges, grad_x, grad_y, grad_x_e, grad_y_e
    

def poisson_node(mesh, boundaries, init=default_init):
    # Only Dirichlet bnd for now
    
    
    cells = mesh['cells']
    nodes = mesh['nodes']
    volumes = nodes['volumes']
    pairs = mesh['pairs']
    n_cells = cells['n']
    n_pairs = pairs['n']
    n_nodes = nodes['n']
    
    phi = np.zeros(n_nodes)
    phi_edges = np.zeros(n_pairs)
    grad_x = np.zeros(n_nodes)
    grad_y = np.zeros(n_nodes)
    grad_x_e = np.zeros(n_pairs)
    grad_y_e = np.zeros(n_pairs)
    
    matrix = np.zeros((n_nodes, n_nodes))
    rhs = np.zeros(n_nodes)
    
    # Init
    for i in range(n_nodes):
        phi[i] = init(nodes['centers'][i][0], nodes['centers'][i][1])

    def compute_grad():
        
        for _ in range(GRAD_ITER_MAX):
            # Face values
            for i in range(n_pairs):
                if pairs['on_bnd'][i]:
                    continue
                node_1 = pairs['nodes'][i][0]
                node_2 = pairs['nodes'][i][1]
                
                phi_edges[i] = (phi[node_1] + phi[node_2])*0.5
                middle_x = (nodes['centers'][node_1][0] + nodes['centers'][node_2][0])/2.
                middle_y = (nodes['centers'][node_1][1] + nodes['centers'][node_2][1])/2.
                phi_edges[i] += 0.5*((grad_x[node_1] + grad_x[node_2])*(pairs['centers'][i][0] - middle_x) + (grad_y[node_1] + grad_y[node_2])*(pairs['centers'][i][1] - middle_y))
                
            for i_bnd, bnd in enumerate(boundaries):
                if bnd[0] == 'D':
                    for edge in mesh['boundaries']['faces'][i_bnd]:
                        # print(edge)
                        # print(bnd[1])
                        phi_edges[edge] = bnd[1]
                else:
                    raise ValueError("Bnd not implemented")
                
            # Grads
            grad_x[:] = 0.
            grad_y[:] = 0.
            
            for i in range(n_pairs):
                area = pairs['nodes_areas'][i]
                flux_x = area * phi_edges[i] * pairs['nodes_normals'][i][0]
                flux_y = area * phi_edges[i] * pairs['nodes_normals'][i][1]
                grad_x[node_1] += flux_x/nodes['volumes'][node_1]
                grad_y[node_1] += flux_y/nodes['volumes'][node_1]
                
                grad_x[node_2] -= flux_x/nodes['volumes'][node_2]
                grad_y[node_2] -= flux_y/nodes['volumes'][node_2]
                
                
                
        for i in range(n_pairs):
            if pairs['on_bnd'][i]:
                continue
            node_1 = pairs['nodes'][i][0]
            node_2 = pairs['nodes'][i][1]
            
            d_cf_x = nodes['centers'][node_2][0] - nodes['centers'][node_1][0]
            d_cf_y = nodes['centers'][node_2][1] - nodes['centers'][node_1][1]
            d_cf_norm = np.sqrt(d_cf_x**2 + d_cf_y**2)
            e_cf_x = d_cf_x/d_cf_norm
            e_cf_y = d_cf_y/d_cf_norm
    
            gw = np.sqrt((pairs['centers'][i][0] - nodes['centers'][node_2][0])**2 + (pairs['centers'][i][1] - nodes['centers'][node_2][1])**2)/d_cf_norm
            
            grad_x_e[i] = grad_x[node_1]*gw + grad_x[node_2]*(1.-gw)
            grad_y_e[i] = grad_y[node_1]*gw + grad_y[node_2]*(1.-gw)
            
            grad_x_e[i] = grad_x_e[i] + e_cf_x*((phi[node_2] - phi[node_1])/d_cf_norm - (grad_x_e[i]*e_cf_x + grad_y_e[i]*e_cf_y))
            grad_y_e[i] = grad_y_e[i] + e_cf_y*((phi[node_2] - phi[node_1])/d_cf_norm - (grad_x_e[i]*e_cf_x + grad_y_e[i]*e_cf_y))
            
        for i_bnd, bnd in enumerate(boundaries):
            if bnd[0] == 'D':
                for i_edge, edge in enumerate(mesh['boundaries']['faces'][i_bnd]):
                    node_1 = pairs['nodes'][edge][0]
                    node_2 = pairs['nodes'][edge][1]
                    
                    d_cf_x = nodes['centers'][node_2][0] - nodes['centers'][node_1][0]
                    d_cf_y = nodes['centers'][node_2][1] - nodes['centers'][node_1][1]
                    d_cf_norm = np.sqrt(d_cf_x**2 + d_cf_y**2)
                    e_cf_x = d_cf_x/d_cf_norm
                    e_cf_y = d_cf_y/d_cf_norm
            
                    gw = np.sqrt((pairs['centers'][edge][0] - nodes['centers'][node_2][0])**2 + (pairs['centers'][edge][1] - nodes['centers'][node_2][1])**2)/d_cf_norm
                    
                    grad_x_e[edge] = grad_x[node_1]*gw + grad_x[node_2]*(1.-gw)
                    grad_y_e[edge] = grad_y[node_1]*gw + grad_y[node_2]*(1.-gw)
                    
                    grad_x_e[edge] = grad_x_e[edge] + e_cf_x*((phi[node_2] - phi[node_1])/d_cf_norm - (grad_x_e[i]*e_cf_x + grad_y_e[i]*e_cf_y))
                    grad_y_e[edge] = grad_y_e[edge] + e_cf_y*((phi[node_2] - phi[node_1])/d_cf_norm - (grad_x_e[i]*e_cf_x + grad_y_e[i]*e_cf_y))
            else:
                raise ValueError("Bnd not implemented")
        return
    
    def laplacian():
        matrix[:, :] = 0.
        rhs[:] = 0.
        
        for i in range(n_pairs):
            if pairs['on_bnd'][i]:
                continue
            node_1 = pairs['nodes'][i][0]
            node_2 = pairs['nodes'][i][1]
            
            s = pairs['nodes_areas'][i]
            
            e_f_x = nodes['centers'][node_2][0] - nodes['centers'][node_1][0]
            e_f_y = nodes['centers'][node_2][1] - nodes['centers'][node_1][1]
            d_cf_norm = np.sqrt(e_f_x**2 + e_f_y**2)
            e_f_x *= s/d_cf_norm
            e_f_y *= s/d_cf_norm
            s_f_x = pairs['nodes_normals'][i][0]*s
            s_f_y = pairs['nodes_normals'][i][1]*s
            t_f_x = s_f_x - e_f_x
            t_f_y = s_f_y - e_f_y
            
            flux_f = - s / d_cf_norm
            
            matrix[node_1, node_1] -= flux_f
            matrix[node_1, node_2] += flux_f
            
            matrix[node_2, node_2] -= flux_f
            matrix[node_2, node_1] += flux_f
            
            rhs[node_1] -= grad_x_e[i] * t_f_x + grad_y_e[i] * t_f_y
            rhs[node_2] += grad_x_e[i] * t_f_x + grad_y_e[i] * t_f_y
            
        for i_bnd, bnd in enumerate(boundaries):
            if bnd[0] == 'D':
                for node in mesh['boundaries']['nodes'][i_bnd]:
                    
                    matrix[node, :] = 0.
                    matrix[node, node] = 1.
                    
                    rhs[node] = bnd[1]
            else:
                raise ValueError("Bnd not implemented")
        
    # for i in range(4):
    #     compute_grad()
        
    #     laplacian()
        
    #     phi = np.linalg.solve(matrix, rhs)
    
    for _ in range(20):
        compute_grad()
            
        laplacian()
        phi = jacobi(matrix, rhs, phi)
        
    compute_grad()
    
    
    return phi, phi_edges, grad_x, grad_y, grad_x_e, grad_y_e