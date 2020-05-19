# TODO: title

# Done:
Optimization settings:
- true angles and predicted angles (without a flip)
- optimizer = Adam
- learning rate = 1e-02
- batch size = 256
- steps = 300
- m = [1, 1, 1, 1] no flip

Goal: estimate 6D vector for 4D alignment rotation

- [x] Sometimes it converges sometimes not: why?  
  - trajectories of both cases collected with visualization of the solution space
  - fixed last 3 elements of the 6D vecor (angle_alignment.ipynb) and iterating through solution losses of the first 3 dimensions 
  - embedding 6D vector to 2D space (projections.ipynb) using PCA, MDS, TSNA, Isomap manifold learning methods and collecting trajectories
  
- [x] There is no unique solution for this optimization mehtod
  - method to compare if 2 rotations are the same
  - method for grouping same rotations (finding unique angle representative)
  - 
- [x] Different result visualizations to visualize the alignment of angles
  - visualize optimization iterations on the polar plot
  - visualize optimization iterations on the rotation vector plot
  
- [x] Solver for a 4D rotation matrix
  - wasn't able to calculate unique solution for 16 equations
  
# Todo:
- [ ] project title
- [ ] Test different optimizers (with e.g. 1e-02 learning rate)
  - Adam vs. SGD vs. RMSProp vs. Stochastic Frank Wolfe vs. etc.
  - make trajectory comparison on one plot with the same starting point
- [ ] Test different learning rates (with e.g. Adam)
  - 1e-02 current, add several others 
  - make trajectory comparison on one plot with the same starting point
- [ ] Explore the solution of transposed angles and check why it never converges to optimum (therefore we need a flip)

# Misc theory
- Z-Y-Z rotations: explain what each angle means
- quaternions and $S^3$ space
- conversion: euler angle <-> quaternion
- $d_q$: distanece between 2 quaternions
- 4D rotation in SO(4) equations, explain why not 3D rotation
- Transposed angles: if we have $R = R_z(\alpha)R_y(\beta)R_z(\gamma)$ where $R$ is 3D matrix, then it's transpose can be obtained with following angles $R^T = R_z(-\gamma)R_y(-\beta)R_z(-\alpha)$.
- Flip (global reflection): $R = R^T * [-1, -1, -1, 1]$

