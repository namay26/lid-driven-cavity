'''
    We will be using projection method to calculate the velocities of the points in a mesh grid (41x41) of a lid-driven cavity. The caivty 
    (Must use boundary conditions)
    For this we have a three step approach : 
    
        1) del_u/del_t + (u ⋅ ∇) u = ν ∇²u  # Solving the momentum equation without pressure gradient. The velocity obtained are not correct so this is basically a predictive step which needs a corrector.
           For this, we discretise using the FTCS scheme. (Forward in Time and Central in Space)
        2) ∇²p = ρ/Δt ∇ ⋅ u # Solve this possion equation for the pressure. This will be needed for the corrective step.
        3) u_corr = u+pred − Δt/ρ ∇ p  # Correct the velocities and then use the boundary conditions.

'''

import matplotlib.pyplot as plt
import numpy as np

# Initialising Constants

N_GRID_POINTS = 41
DOMAIN_SIZE = 1.0
NUM_ITERATIONS = 500
TIME_STEP = 0.001
REYNOLDS_NUMBER = 10
DENSITY = 1.0
HORIZONTAL_VEL_TOP = 1.0

NUM_PRESSURE_POISSON_EQN = 50

VISCOSITY = DOMAIN_SIZE * HORIZONTAL_VEL_TOP / REYNOLDS_NUMBER

def main():
    element_len = DOMAIN_SIZE / (N_GRID_POINTS - 1)
    x = np.linspace(0, DOMAIN_SIZE, N_GRID_POINTS)
    y = np.linspace(0, DOMAIN_SIZE, N_GRID_POINTS)
    
    X,Y = np.meshgrid (x,y)
    
    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)
    
    def CDS_X(k):
        diff = np.zeros_like(k)
        diff[1:-1, 1:-1] = (k[1:-1, 2:  ] - k[1:-1, 0:-2]) / (2 * element_len) # d()/dx = (u(i+1,j)-u(i-1,j))/2*del_x
        return diff
    
    def CDS_Y(k):
        diff = np.zeros_like(k)
        diff[1:-1, 1:-1] = (k[2: , 1:-1] - k[0:-2, 1:-1]) / (2 * element_len) # d()/dY = (u(i,j+1)-u(i,j-1))/2*del_y
        return diff
        
    def laplace(k):
        diff = np.zeros_like(k)
        diff[1:-1, 1:-1] = (k[1:-1, 0:-2] + k[0:-2, 1:-1] - 4 * k[1:-1, 1:-1] + k[1:-1, 2:  ] + k[2:  , 1:-1]) / (element_len**2) 
        return diff
    
    for _ in range(NUM_ITERATIONS):
        duprev_dx = CDS_X(u_prev)
        duprev_dy = CDS_Y(u_prev)
        dvprev_dx = CDS_X(v_prev)
        dvprev_dy = CDS_Y(v_prev)
        laplace_uprev = laplace(u_prev)
        laplace_vprev = laplace(v_prev)
        
        # Now for each iteration let's perform the steps
        # 1) Predictive Steps
        u_pred = u_prev + TIME_STEP * (( VISCOSITY * laplace_uprev ) - (u_prev * duprev_dx + v_prev * duprev_dy))
        v_pred = v_prev + TIME_STEP * (( VISCOSITY * laplace_vprev ) - (u_prev * dvprev_dx + v_prev * dvprev_dy))   
       
        # Applying Boundary Conditions 
        u_pred[0, :] = 0.0
        u_pred[:, 0] = 0.0
        u_pred[:, -1] = 0.0
        u_pred[-1, :] = HORIZONTAL_VEL_TOP 
        v_pred[0, :] = 0.0
        v_pred[:, 0] = 0.0
        v_pred[:, -1] = 0.0
        v_pred[-1, :] = 0.0
        
        dupred_dx = CDS_X(u_pred)
        dvpred_dy = CDS_Y(v_pred)
        
        # Computing the pressule laplacian
        rhs = DENSITY / TIME_STEP * (dupred_dx + dvpred_dy)
        
        for _ in range (NUM_PRESSURE_POISSON_EQN):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (p_prev[1:-1, 0:-2] + p_prev[0:-2, 1:-1] + p_prev[1:-1, 2:  ] + p_prev[2:  , 1:-1] - element_len**2 * rhs[1:-1, 1:-1])
            
            # Applying Boundary Conditions to pressure terms
            p_next[:, -1] = p_next[:, -2]
            p_next[0,  :] = p_next[1,  :]
            p_next[:,  0] = p_next[:,  1]
            p_next[-1, :] = 0.0

            p_prev = p_next
                    
        # Correcting the predicted velocities
        dpnext_dx = CDS_X(p_next)
        dpnext_dy = CDS_Y(p_next)
        
        u_next = (
            u_pred
            -
            TIME_STEP / DENSITY
            *
            dpnext_dx
        )
        v_next = (
            v_pred
            -
            TIME_STEP / DENSITY
            *
            dpnext_dy
        )
        
        # Applying Boundary Conditions to the velocity terms
        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = HORIZONTAL_VEL_TOP
        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0
        
        
        # Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next
        

    plt.style.use("dark_background")
    plt.figure()
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2], cmap="coolwarm")
    plt.colorbar()

    plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()


if __name__ == "__main__":
    main()
