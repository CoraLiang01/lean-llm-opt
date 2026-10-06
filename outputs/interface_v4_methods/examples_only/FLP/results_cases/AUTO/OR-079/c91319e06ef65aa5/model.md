Let us define the following sets, parameters, and variables based on the provided CSV data:

Sets:
- \( I = \{A1, A2, ..., A15\} \): Set of potential factory sites.
- \( J = \{B1, B2, ..., B8\} \): Set of distribution centers.

Parameters:
- \( f_i \): Fixed cost of opening factory \( i \in I \).
- \( K_i \): Capacity of factory \( i \in I \).
- \( d_j \): Demand at distribution center \( j \in J \).
- \( c_{ij} \): Shipping cost per unit from factory \( i \) to distribution center \( j \).

From facility_costs.csv:
\[
\begin{align*}
&f_{A1} = 0,\quad K_{A1} = 30 \\
&f_{A2} = 175,\quad K_{A2} = 10 \\
&f_{A3} = 300,\quad K_{A3} = 20 \\
&f_{A4} = 375,\quad K_{A4} = 30 \\
&f_{A5} = 500,\quad K_{A5} = 40 \\
&f_{A6} = 200,\quad K_{A6} = 20 \\
&f_{A7} = 260,\quad K_{A7} = 25 \\
&f_{A8} = 220,\quad K_{A8} = 30 \\
&f_{A9} = 320,\quad K_{A9} = 35 \\
&f_{A10} = 280,\quad K_{A10} = 20 \\
&f_{A11} = 350,\quad K_{A11} = 40 \\
&f_{A12} = 420,\quad K_{A12} = 25 \\
&f_{A13} = 470,\quad K_{A13} = 30 \\
&f_{A14} = 520,\quad K_{A14} = 50 \\
&f_{A15} = 560,\quad K_{A15} = 45 \\
\end{align*}
\]

From demand_requirements.csv:
\[
\begin{align*}
&d_{B1} = 30 \\
&d_{B2} = 25 \\
&d_{B3} = 20 \\
&d_{B4} = 35 \\
&d_{B5} = 25 \\
&d_{B6} = 30 \\
&d_{B7} = 25 \\
&d_{B8} = 30 \\
\end{align*}
\]

From shipping_costs.csv, the cost matrix \( c_{ij} \) is:

\[
\begin{array}{c|cccccccc}
      & B1 & B2 & B3 & B4 & B5 & B6 & B7 & B8 \\
\hline
A1   & 8  & 4  & 3  & 6  & 7  & 5  & 9  & 8  \\
A2   & 5  & 2  & 3  & 5  & 6  & 4  & 7  & 6  \\
A3   & 4  & 3  & 4  & 6  & 5  & 5  & 6  & 7  \\
A4   & 9  & 7  & 5  & 8  & 9  & 6  & 10 & 7  \\
A5   & 10 & 4  & 2  & 6  & 8  & 5  & 7  & 3  \\
A6   & 6  & 5  & 4  & 5  & 7  & 6  & 8  & 5  \\
A7   & 7  & 6  & 5  & 4  & 6  & 7  & 9  & 6  \\
A8   & 5  & 4  & 6  & 3  & 5  & 6  & 7  & 6  \\
A9   & 8  & 7  & 6  & 7  & 9  & 8  & 10 & 7  \\
A10  & 6  & 5  & 7  & 4  & 6  & 5  & 7  & 5  \\
A11  & 9  & 6  & 4  & 6  & 8  & 7  & 9  & 6  \\
A12  & 7  & 5  & 6  & 5  & 6  & 5  & 8  & 5  \\
A13  & 8  & 6  & 5  & 6  & 7  & 6  & 8  & 7  \\
A14  & 9  & 5  & 3  & 5  & 7  & 4  & 6  & 4  \\
A15  & 10 & 6  & 4  & 5  & 8  & 5  & 7  & 5  \\
\end{array}
\]

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if factory \( i \) is constructed, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount shipped from factory \( i \) to distribution center \( j \).

Mathematical Model:

\[
\begin{align*}
\textbf{Objective:} \quad & \min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\\
\textbf{Subject to:} \\
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \quad \text{(meet demand at each DC)} \\
& \sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I \quad \text{(do not exceed capacity; only if open)} \\
& x_{ij} \geq 0 \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
\end{align*}
\]

Where all parameters (\( f_i, K_i, d_j, c_{ij} \)) are as listed above, directly from the CSV files.

This model finds the subset of factories to open and the shipment plan from open factories to distribution centers that minimizes the total system cost (fixed + variable), while meeting all demand and respecting facility capacities.