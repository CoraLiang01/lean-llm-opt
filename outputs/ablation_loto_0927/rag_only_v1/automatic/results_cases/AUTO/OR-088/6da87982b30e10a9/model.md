Let:
- \( I = \{1,2,\ldots,15\} \) index set of plants (F1–F15)
- \( J = \{1,2,\ldots,15\} \) index set of customers (C1–C15)

Parameters:
- \( f_i \): Fixed opening cost for plant \( i \)
- \( K_i \): Capacity of plant \( i \)
- \( c_{ij} \): Per-unit transport cost from plant \( i \) to customer \( j \)
- \( d_j \): Demand of customer \( j \)

Decision variables:
- \( y_i \in \{0,1\} \): 1 if plant \( i \) is opened, 0 otherwise
- \( x_{ij} \geq 0 \): Amount shipped from plant \( i \) to customer \( j \)

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

Parameters (explicitly listed):

- Plant indices: \( i = 1 \) (F1), \( i = 2 \) (F2), ..., \( i = 15 \) (F15)
- Customer indices: \( j = 1 \) (C1), \( j = 2 \) (C2), ..., \( j = 15 \) (C15)

Fixed opening costs (\( f_i \)), capacities (\( K_i \)), and per-unit transport costs (\( c_{ij} \)):

\[
\begin{array}{cccccccccccccccc}
\text{Plant} & f_i & K_i & c_{i1} & c_{i2} & c_{i3} & c_{i4} & c_{i5} & c_{i6} & c_{i7} & c_{i8} & c_{i9} & c_{i10} & c_{i11} & c_{i12} & c_{i13} & c_{i14} & c_{i15} \\
\hline
\text{F1}  & 11250 & 101 & 7.8 & 7.6 & 6.7 & 7.9 & 8.1 & 8.3 & 7.3 & 8.2 & 8.1 & 8.2 & 7.3 & 7.7 & 6.7 & 7.1 & 7.9 \\
\text{F2}  & 13480 & 124 & 5.3 & 6.0 & 5.0 & 6.4 & 5.9 & 6.2 & 5.6 & 6.1 & 6.3 & 6.1 & 5.0 & 5.6 & 5.3 & 4.9 & 6.3 \\
\text{F3}  & 14870 & 139 & 7.2 & 8.1 & 7.4 & 8.8 & 8.5 & 8.7 & 7.7 & 8.7 & 8.9 & 8.5 & 7.2 & 7.7 & 7.1 & 7.6 & 8.4 \\
\text{F4}  & 10290 & 86  & 7.0 & 7.1 & 6.5 & 7.9 & 7.4 & 7.7 & 6.7 & 7.9 & 7.8 & 7.3 & 6.8 & 7.0 & 6.5 & 6.7 & 7.6 \\
\text{F5}  & 16740 & 157 & 3.5 & 3.8 & 2.9 & 4.3 & 3.6 & 3.9 & 3.2 & 4.3 & 4.5 & 4.0 & 3.2 & 4.0 & 2.9 & 3.4 & 3.9 \\
\text{F6}  & 13960 & 133 & 8.2 & 8.6 & 7.9 & 9.5 & 8.5 & 9.3 & 8.5 & 9.4 & 9.0 & 9.2 & 8.1 & 8.7 & 7.9 & 8.5 & 9.0 \\
\text{F7}  & 12680 & 118 & 6.9 & 7.6 & 6.8 & 8.4 & 8.0 & 8.0 & 7.6 & 8.0 & 8.1 & 7.8 & 6.9 & 7.1 & 7.0 & 6.9 & 7.5 \\
\text{F8}  & 17890 & 162 & 6.9 & 7.8 & 7.1 & 8.7 & 8.6 & 8.2 & 7.2 & 7.9 & 8.4 & 7.9 & 7.0 & 7.4 & 6.8 & 7.3 & 8.0 \\
\text{F9}  & 10950 & 92  & 3.5 & 3.8 & 2.8 & 4.4 & 4.2 & 4.8 & 3.8 & 5.0 & 4.5 & 4.1 & 3.2 & 3.7 & 3.7 & 3.2 & 4.5 \\
\text{F10} & 15320 & 144 & 5.2 & 6.1 & 5.1 & 6.3 & 6.1 & 6.0 & 5.6 & 6.5 & 6.2 & 5.9 & 5.3 & 6.1 & 5.1 & 5.2 & 6.2 \\
\text{F11} & 11830 & 107 & 5.2 & 5.5 & 4.5 & 6.2 & 5.7 & 6.1 & 5.1 & 5.8 & 5.7 & 6.2 & 5.2 & 5.2 & 4.5 & 5.1 & 5.4 \\
\text{F12} & 14110 & 129 & 7.8 & 8.7 & 7.6 & 9.0 & 8.6 & 9.0 & 8.5 & 9.3 & 9.3 & 8.4 & 7.9 & 8.2 & 7.4 & 7.6 & 8.7 \\
\text{F13} & 15970 & 151 & 6.7 & 6.6 & 6.1 & 7.3 & 7.1 & 7.5 & 6.7 & 8.0 & 7.6 & 7.2 & 6.3 & 6.9 & 6.2 & 6.0 & 7.2 \\
\text{F14} & 13140 & 113 & 7.5 & 8.6 & 7.6 & 8.2 & 8.0 & 7.9 & 7.5 & 8.7 & 8.8 & 8.1 & 7.2 & 7.3 & 7.0 & 7.0 & 8.0 \\
\text{F15} & 10580 & 85  & 5.1 & 5.8 & 4.6 & 5.9 & 6.5 & 5.9 & 5.2 & 7.0 & 7.1 & 5.9 & 5.1 & 5.8 & 5.4 & 4.9 & 6.0 \\
\end{array}
\]

Customer demands (\( d_j \)):

\[
\begin{array}{cc}
\text{Customer} & d_j \\
\hline
\text{C1}  & 83 \\
\text{C2}  & 76 \\
\text{C3}  & 91 \\
\text{C4}  & 68 \\
\text{C5}  & 104 \\
\text{C6}  & 97 \\
\text{C7}  & 88 \\
\text{C8}  & 73 \\
\text{C9}  & 109 \\
\text{C10} & 95 \\
\text{C11} & 82 \\
\text{C12} & 67 \\
\text{C13} & 113 \\
\text{C14} & 79 \\
\text{C15} & 92 \\
\end{array}
\]

Summary:

Minimize:
\[
\sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{15} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i=1}^{15} x_{ij} = d_j \quad \forall j = 1,\ldots,15
\]
\[
\sum_{j=1}^{15} x_{ij} \leq K_i y_i \quad \forall i = 1,\ldots,15
\]
\[
y_i \in \{0,1\}, \quad x_{ij} \geq 0
\]

All parameter values are as listed above, directly from the CSV files.