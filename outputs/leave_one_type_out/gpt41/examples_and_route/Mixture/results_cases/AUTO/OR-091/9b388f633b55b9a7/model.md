Let $x_i$ be a binary variable indicating whether Operations Research course $i$ is selected ($x_i = 1$) or not ($x_i = 0$).

Courses (in source order):

| course_id | course_name                              | credits | interest_points |
|-----------|------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming  | 5       | 95              |
| C23       | Integer Programming                      | 5       | 92              |
| C24       | Stochastic Processes                     | 4       | 86              |
| C25       | Simulation Modeling                      | 4       | 82              |
| C26       | Network Flows                            | 4       | 85              |
| C27       | Queueing Theory                          | 4       | 80              |
| C28       | Revenue Management                       | 4       | 88              |

Objective:
$$
\max \; 95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}
$$

Subject to:
$$
5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20
$$

$$
x_{22}, x_{23}, x_{24}, x_{25}, x_{26}, x_{27}, x_{28} \in \{0,1\}
$$

Where:
- $x_{22}$: Operations Research: Linear Programming
- $x_{23}$: Integer Programming
- $x_{24}$: Stochastic Processes
- $x_{25}$: Simulation Modeling
- $x_{26}$: Network Flows
- $x_{27}$: Queueing Theory
- $x_{28}$: Revenue Management

All variables and coefficients are as retrieved, in source order.