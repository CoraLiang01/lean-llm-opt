Let $I$ be the set of Operations Research courses:
$$
I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
$$

Define binary decision variables:
$$
x_i = \begin{cases}
1 & \text{if course } i \text{ is selected} \\
0 & \text{otherwise}
\end{cases}
\quad \forall i \in I
$$

Parameters (from source order):

\[
\begin{array}{llll}
\text{Course} & \text{Credits } (c_i) & \text{Interest Points } (p_i) \\
\hline
\text{C22} & 5 & 95 \\
\text{C23} & 5 & 92 \\
\text{C24} & 4 & 86 \\
\text{C25} & 4 & 82 \\
\text{C26} & 4 & 85 \\
\text{C27} & 4 & 80 \\
\text{C28} & 4 & 88 \\
\end{array}
\]

Objective:
$$
\max \sum_{i \in I} p_i x_i = 95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}
$$

Subject to:
\[
\sum_{i \in I} c_i x_i \leq 20
\]
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

Explicitly:
\[
5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20
\]
\[
x_{22}, x_{23}, x_{24}, x_{25}, x_{26}, x_{27}, x_{28} \in \{0,1\}
\]