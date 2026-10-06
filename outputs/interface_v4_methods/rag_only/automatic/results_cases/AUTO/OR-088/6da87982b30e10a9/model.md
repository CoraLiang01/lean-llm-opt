Let:
- \( I = \{1,2,\ldots,15\} \) index the plants, corresponding to F1–F15.
- \( J = \{1,2,\ldots,15\} \) index the customers, corresponding to C1–C15.

Parameters:
- \( f_i \): Fixed opening cost for plant \( i \).
- \( K_i \): Capacity of plant \( i \).
- \( c_{ij} \): Per-unit transport cost from plant \( i \) to customer \( j \).
- \( d_j \): Demand of customer \( j \).

Variables:
- \( y_i \in \{0,1\} \): 1 if plant \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount shipped from plant \( i \) to customer \( j \).

Objective:
\[
\min \sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{15} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i=1}^{15} x_{ij} = d_j \quad \forall j = 1,\ldots,15
\]
2. Capacity constraint for each plant:
\[
\sum_{j=1}^{15} x_{ij} \leq K_i y_i \quad \forall i = 1,\ldots,15
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i = 1,\ldots,15
\]
\[
x_{ij} \geq 0 \quad \forall i = 1,\ldots,15; \; j = 1,\ldots,15
\]

Explicit parameter values:

- Fixed opening costs (\( f_i \)), capacities (\( K_i \)), and per-unit transport costs (\( c_{ij} \)):

\[
\begin{array}{l|l|l|ccccccccccccccc}
\text{Plant} & f_i & K_i & c_{i1} & c_{i2} & c_{i3} & c_{i4} & c_{i5} & c_{i6} & c_{i7} & c_{i8} & c_{i9} & c_{i10} & c_{i11} & c_{i12} & c_{i13} & c_{i14} & c_{i15} \\
\hline
F1  & 11250 & 101 & 7.8 & 7.6 & 6.7 & 7.9 & 8.1 & 8.3 & 7.3 & 8.2 & 8.1 & 8.2 & 7.3 & 7.7 & 6.7 & 7.1 & 7.9 \\
F2  & 13480 & 124 & 5.3 & 6.0 & 5.0 & 6.4 & 5.9 & 6.2 & 5.6 & 6.1 & 6.3 & 6.1 & 5.0 & 5.6 & 5.3 & 4.9 & 6.3 \\
F3  & 14870 & 139 & 7.2 & 8.1 & 7.4 & 8.8 & 8.5 & 8.7 & 7.7 & 8.7 & 8.9 & 8.5 & 7.2 & 7.7 & 7.1 & 7.6 & 8.4 \\
F4  & 10290 & 86  & 7.0 & 7.1 & 6.5 & 7.9 & 7.4 & 7.7 & 6.7 & 7.9 & 7.8 & 7.3 & 6.8 & 7.0 & 6.5 & 6.7 & 7.6 \\
F5  & 16740 & 157 & 3.5 & 3.8 & 2.9 & 4.3 & 3.6 & 3.9 & 3.2 & 4.3 & 4.5 & 4.0 & 3.2 & 4.0 & 2.9 & 3.4 & 3.9 \\
F6  & 13960 & 133 & 8.2 & 8.6 & 7.9 & 9.5 & 8.5 & 9.3 & 8.5 & 9.4 & 9.0 & 9.2 & 8.1 & 8.7 & 7.9 & 8.5 & 9.0 \\
F7  & 12680 & 118 & 6.9 & 7.6 & 6.8 & 8.4 & 8.0 & 8.0 & 7.6 & 8.0 & 8.1 & 7.8 & 6.9 & 7.1 & 7.0 & 6.9 & 7.5 \\
F8  & 17890 & 162 & 6.9 & 7.8 & 7.1 & 8.7 & 8.6 & 8.2 & 7.2 & 7.9 & 8.4 & 7.9 & 7.0 & 7.4 & 6.8 & 7.3 & 8.0 \\
F9  & 10950 & 92  & 3.5 & 3.8 & 2.8 & 4.4 & 4.2 & 4.8 & 3.8 & 5.0 & 4.5 & 4.1 & 3.2 & 3.7 & 3.7 & 3.2 & 4.5 \\
F10 & 15320 & 144 & 5.2 & 6.1 & 5.1 & 6.3 & 6.1 & 6.0 & 5.6 & 6.5 & 6.2 & 5.9 & 5.3 & 6.1 & 5.1 & 5.2 & 6.2 \\
F11 & 11830 & 107 & 5.2 & 5.5 & 4.5 & 6.2 & 5.7 & 6.1 & 5.1 & 5.8 & 5.7 & 6.2 & 5.2 & 5.2 & 4.5 & 5.1 & 5.4 \\
F12 & 14110 & 129 & 7.8 & 8.7 & 7.6 & 9.0 & 8.6 & 9.0 & 8.5 & 9.3 & 9.3 & 8.4 & 7.9 & 8.2 & 7.4 & 7.6 & 8.7 \\
F13 & 15970 & 151 & 6.7 & 6.6 & 6.1 & 7.3 & 7.1 & 7.5 & 6.7 & 8.0 & 7.6 & 7.2 & 6.3 & 6.9 & 6.2 & 6.0 & 7.2 \\
F14 & 13140 & 113 & 7.5 & 8.6 & 7.6 & 8.2 & 8.0 & 7.9 & 7.5 & 8.7 & 8.8 & 8.1 & 7.2 & 7.3 & 7.0 & 7.0 & 8.0 \\
F15 & 10580 & 85  & 5.1 & 5.8 & 4.6 & 5.9 & 6.5 & 5.9 & 5.2 & 7.0 & 7.1 & 5.9 & 5.1 & 5.8 & 5.4 & 4.9 & 6.0 \\
\end{array}
\]

- Customer demands (\( d_j \)):

\[
\begin{array}{l|ccccccccccccccc}
\text{Customer} & d_1 & d_2 & d_3 & d_4 & d_5 & d_6 & d_7 & d_8 & d_9 & d_{10} & d_{11} & d_{12} & d_{13} & d_{14} & d_{15} \\
\hline
\text{Value} & 83 & 76 & 91 & 68 & 104 & 97 & 88 & 73 & 109 & 95 & 82 & 67 & 113 & 79 & 92 \\
\end{array}
\]

Summary of the full model:

\[
\boxed{
\begin{align*}
\min \quad & \sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{15} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i=1}^{15} x_{ij} = d_j \quad \forall j = 1,\ldots,15 \\
& \sum_{j=1}^{15} x_{ij} \leq K_i y_i \quad \forall i = 1,\ldots,15 \\
& y_i \in \{0,1\} \quad \forall i = 1,\ldots,15 \\
& x_{ij} \geq 0 \quad \forall i = 1,\ldots,15; \; j = 1,\ldots,15
\end{align*}
}
\]

with all parameter values as listed above.