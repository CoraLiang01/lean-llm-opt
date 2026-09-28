##### Decision Variables

For each Operations Research course $i\in\{C22, C23, C24, C25, C26, C27, C28\}$:
$$
x_i \in \{0,1\}
$$
where $x_i=1$ if course $i$ is selected, $0$ otherwise.

##### Parameters

\[
\begin{array}{llll}
\text{Course} & \text{Credits } (c_i) & \text{Interest Points } (p_i) \\
\hline
C22 & 5 & 95 \\
C23 & 5 & 92 \\
C24 & 4 & 86 \\
C25 & 4 & 82 \\
C26 & 4 & 85 \\
C27 & 4 & 80 \\
C28 & 4 & 88 \\
\end{array}
\]

##### Objective Function

\[
\max \; 95x_{C22} + 92x_{C23} + 86x_{C24} + 82x_{C25} + 85x_{C26} + 80x_{C27} + 88x_{C28}
\]

##### Constraint

\[
5x_{C22} + 5x_{C23} + 4x_{C24} + 4x_{C25} + 4x_{C26} + 4x_{C27} + 4x_{C28} \leq 20
\]

\[
x_i \in \{0,1\} \quad \forall i \in \{C22, C23, C24, C25, C26, C27, C28\}
\]