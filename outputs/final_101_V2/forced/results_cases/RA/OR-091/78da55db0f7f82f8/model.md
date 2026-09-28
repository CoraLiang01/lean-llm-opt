Let $x_i$ be a binary variable indicating whether to enroll in Operations Research course $i$ ($x_i = 1$) or not ($x_i = 0$).

Let the set of Operations Research courses (in source order) be:

- C22: Operations Research: Linear Programming (5 credits, 95 interest points)
- C23: Integer Programming (5 credits, 92 interest points)
- C24: Stochastic Processes (4 credits, 86 interest points)
- C25: Simulation Modeling (4 credits, 82 interest points)
- C26: Network Flows (4 credits, 85 interest points)
- C27: Queueing Theory (4 credits, 80 interest points)
- C28: Revenue Management (4 credits, 88 interest points)

Objective:
$$
\max \; 95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}
$$

Subject to:
$$
5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20
$$

$$
x_{i} \in \{0,1\} \quad \text{for } i \in \{22,23,24,25,26,27,28\}
$$

Where:
- $x_{22}$: Operations Research: Linear Programming
- $x_{23}$: Integer Programming
- $x_{24}$: Stochastic Processes
- $x_{25}$: Simulation Modeling
- $x_{26}$: Network Flows
- $x_{27}$: Queueing Theory
- $x_{28}$: Revenue Management

All coefficients and course IDs are as retrieved, in source order.