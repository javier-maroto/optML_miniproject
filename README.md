# Exploring the Solution Space of Aligning Two Sets of 3D Rotation Angles

The goal of our project is to explore the performance of different algorithms on our non-convex problem: aligning two sets of 3-dimensional rotation angles. The current results obtained on different combination of settings used and experimental conditions can be found in this repository.

## Report
More details on experiments performed and implementation can be found in the [report]().

## Getting Started
Download and install [Anaconda](https://www.anaconda.com/products/individual) on your machine. Note: Python 3.6+.
Then open the terminal and do following:
```
# clone the repo
$ https://github.com/javier-maroto/optML_miniproject.git

# install git lfs support for files
$ git lfs install
# download the files
$ git lfs fetch
# updates the files in the local directory
$ git lfs pull

# position yourself inside the project
$ cd optML_miniproject

# create environment
$ conda env create -f environment.yml

# activate environment
$ conda activate angle_alignment
```

## Run the Experiment
To run the experiment that compares the performance of different optimizers use the following script:
```
$ python main.py
```
Other experiments can be found in the following notebooks:
1. [Discovering non-convexity](https://nbviewer.jupyter.org/github/javier-maroto/optML_miniproject/blob/master/notebooks/angle_alignment_nonconvex.ipynb)
2. [Limited solution space visualizations](https://nbviewer.jupyter.org/github/javier-maroto/optML_miniproject/blob/master/notebooks/angle_alignment_limited_solution_space.ipynb)
3. [Proposal of another optimization equation](https://nbviewer.jupyter.org/github/javier-maroto/optML_miniproject/blob/master/notebooks/angle_alignment_another_equation_proposal.ipynb)
4. [Critical point analysis](https://nbviewer.jupyter.org/github/javier-maroto/optML_miniproject/blob/master/notebooks/critical_point_analysis.ipynb)
5. [Test rotation similarity and finding unique angles](https://nbviewer.jupyter.org/github/javier-maroto/optML_miniproject/blob/master/notebooks/test_rotation_similarity_and_unique_angle.ipynb)


## Team
- [Javier Alejandro Maroto Morales](https://people.epfl.ch/javier.marotomorales/?lang=en), javier.marotomorales@epfl.ch
- [Jelena Banjac](https://jelenabanjac.com/), jelena.banjac@epfl.ch

## Internal Milestones
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
  
- [x] Different result visualizations to visualize the alignment of angles
  - visualize optimization iterations on the polar plot
  - visualize optimization iterations on the rotation vector plot
  
- [x] Solver for a 4D rotation matrix
  - wasn't able to calculate unique solution for 16 equations

- [x] project title
- [x] Test different optimizers (with e.g. 1e-02 learning rate)
  - Adam vs. SGD vs. RMSProp vs. etc.
  - make trajectory comparison on one plot with the same starting point
- [x] Test different learning rates (with e.g. Adam)
  - 1e-02 current, add several others 
  - make trajectory comparison on one plot with the same starting point
- [x] Explore the solution of transposed angles and check why it never converges to optimum (therefore we need a flip)

Technical:
- [x] Organize nitebooks and remove code from them
- [x] Add git LFS support so results can be reproduced


## Loss Variation
Solution space of first 3 vector values visualized in 2D and 3D:
![](./images/solution_space_2D.gif)
![](./images/solution_space3d.gif)
