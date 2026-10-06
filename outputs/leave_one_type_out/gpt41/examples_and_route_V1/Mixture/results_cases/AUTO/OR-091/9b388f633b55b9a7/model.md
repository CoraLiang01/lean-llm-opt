Let $x_i$ be a binary variable indicating whether Operations Research course $i$ is selected ($x_i = 1$) or not ($x_i = 0$).

Let the set of Operations Research courses be:
\[
\begin{array}{lllr}
\text{course\_id} & \text{course\_name} & \text{credits} & \text{interest\_points} \\
\hline
\text{C22} & \text{Operations Research: Linear Programming} & 5 & 95 \\
\text{C23} & \text{Integer Programming} & 5 & 92 \\
\text{C24} & \text{Stochastic Processes} & 4 & 86 \\
\text{C25} & \text{Simulation Modeling} & 4 & 82 \\
\text{C26} & \text{Network Flows} & 4 & 85 \\
\text{C27} & \text{Queueing Theory} & 4 & 80 \\
\text{C28} & \text{Revenue Management} & 4 & 88 \\
\end{array}
\]

Objective:
\[
\max \; 95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}
\]

Subject to:
\[
5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20
\]
\[
x_{i} \in \{0,1\} \quad \text{for } i \in \{22,23,24,25,26,27,28\}
\]

where $x_{22}$ corresponds to course C22, $x_{23}$ to C23, etc.