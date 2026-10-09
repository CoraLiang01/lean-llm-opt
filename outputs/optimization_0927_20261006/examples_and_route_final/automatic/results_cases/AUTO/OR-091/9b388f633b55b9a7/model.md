Let $x_i \in \{0,1\}$ be a binary variable indicating whether Operations Research course $i$ is selected (1) or not (0), for each Operations Research course $i$ in the table below.

Let $I$ be the set of Operations Research courses:

| course_id | course_name                                 | credits | interest_points |
|-----------|---------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming     | 5       | 95              |
| C23       | Integer Programming                        | 5       | 92              |
| C24       | Stochastic Processes                       | 4       | 86              |
| C25       | Simulation Modeling                        | 4       | 82              |
| C26       | Network Flows                              | 4       | 85              |
| C27       | Queueing Theory                            | 4       | 80              |
| C28       | Revenue Management                         | 4       | 88              |

The mathematical model is:

$$
\begin{align*}
\max \quad & 95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28} \\
\text{s.t.} \quad & 5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20 \\
& x_{22}, x_{23}, x_{24}, x_{25}, x_{26}, x_{27}, x_{28} \in \{0,1\}
\end{align*}
$$

Where:
- $x_{22}$: 1 if "Operations Research: Linear Programming" is selected, 0 otherwise
- $x_{23}$: 1 if "Integer Programming" is selected, 0 otherwise
- $x_{24}$: 1 if "Stochastic Processes" is selected, 0 otherwise
- $x_{25}$: 1 if "Simulation Modeling" is selected, 0 otherwise
- $x_{26}$: 1 if "Network Flows" is selected, 0 otherwise
- $x_{27}$: 1 if "Queueing Theory" is selected, 0 otherwise
- $x_{28}$: 1 if "Revenue Management" is selected, 0 otherwise

All other courses are not eligible for selection. The objective is to maximize the total interest points from selected Operations Research courses, subject to a total credit limit of 20. Each course can be selected at most once.