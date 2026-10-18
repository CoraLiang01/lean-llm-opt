Let $x_i$ be a binary variable indicating whether Operations Research course $i$ is selected ($x_i = 1$) or not ($x_i = 0$).

Let $\mathcal{O}$ be the set of Operations Research courses:
\[
\mathcal{O} = \{
\begin{array}{ll}
\text{C22}: & \text{Operations Research: Linear Programming} \\
\text{C23}: & \text{Integer Programming} \\
\text{C24}: & \text{Stochastic Processes} \\
\text{C25}: & \text{Simulation Modeling} \\
\text{C26}: & \text{Network Flows} \\
\text{C27}: & \text{Queueing Theory} \\
\text{C28}: & \text{Revenue Management}
\end{array}
\}
\]

The data for these courses is:

| course_id | course_name                                 | credits | interest_points |
|-----------|---------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming     | 5       | 95              |
| C23       | Integer Programming                        | 5       | 92              |
| C24       | Stochastic Processes                       | 4       | 86              |
| C25       | Simulation Modeling                        | 4       | 82              |
| C26       | Network Flows                              | 4       | 85              |
| C27       | Queueing Theory                            | 4       | 80              |
| C28       | Revenue Management                         | 4       | 88              |

Define variables:
\[
x_i \in \{0,1\} \quad \forall i \in \mathcal{O}
\]

Objective:
\[
\max \sum_{i \in \mathcal{O}} \text{interest\_points}_i \cdot x_i = 95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}
\]

Subject to:
\[
\sum_{i \in \mathcal{O}} \text{credits}_i \cdot x_i \leq 20 \\
5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20
\]
\[
x_i \in \{0,1\} \quad \forall i \in \mathcal{O}
\]

Where:
- $x_{22}$: select Operations Research: Linear Programming (C22)
- $x_{23}$: select Integer Programming (C23)
- $x_{24}$: select Stochastic Processes (C24)
- $x_{25}$: select Simulation Modeling (C25)
- $x_{26}$: select Network Flows (C26)
- $x_{27}$: select Queueing Theory (C27)
- $x_{28}$: select Revenue Management (C28)

All other courses are not eligible for selection.