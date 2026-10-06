Let us define the following sets, parameters, and decision variables:

Sets:
- \( I = \{A1, A2, ..., A15\} \): Set of potential factory sites.
- \( J = \{B1, B2, ..., B8\} \): Set of distribution centers.

Parameters:
- \( f_i \): Fixed cost of opening factory \( i \in I \).
- \( c_{ij} \): Variable shipping cost per unit from factory \( i \) to distribution center \( j \).
- \( d_j \): Demand at distribution center \( j \).
- \( K_i \): Capacity of factory \( i \).

From the CSV data:

Fixed costs and capacities (\( f_i, K_i \)):
- A1: \( f_{A1} = 0 \), \( K_{A1} = 30 \)
- A2: \( f_{A2} = 175 \), \( K_{A2} = 10 \)
- A3: \( f_{A3} = 300 \), \( K_{A3} = 20 \)
- A4: \( f_{A4} = 375 \), \( K_{A4} = 30 \)
- A5: \( f_{A5} = 500 \), \( K_{A5} = 40 \)
- A6: \( f_{A6} = 200 \), \( K_{A6} = 20 \)
- A7: \( f_{A7} = 260 \), \( K_{A7} = 25 \)
- A8: \( f_{A8} = 220 \), \( K_{A8} = 30 \)
- A9: \( f_{A9} = 320 \), \( K_{A9} = 35 \)
- A10: \( f_{A10} = 280 \), \( K_{A10} = 20 \)
- A11: \( f_{A11} = 350 \), \( K_{A11} = 40 \)
- A12: \( f_{A12} = 420 \), \( K_{A12} = 25 \)
- A13: \( f_{A13} = 470 \), \( K_{A13} = 30 \)
- A14: \( f_{A14} = 520 \), \( K_{A14} = 50 \)
- A15: \( f_{A15} = 560 \), \( K_{A15} = 45 \)

Demands (\( d_j \)):
- B1: \( d_{B1} = 30 \)
- B2: \( d_{B2} = 25 \)
- B3: \( d_{B3} = 20 \)
- B4: \( d_{B4} = 35 \)
- B5: \( d_{B5} = 25 \)
- B6: \( d_{B6} = 30 \)
- B7: \( d_{B7} = 25 \)
- B8: \( d_{B8} = 30 \)

Variable shipping costs (\( c_{ij} \)), as a 15x8 matrix:

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
\textbf{Objective:} \quad \min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each distribution center:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Factory capacity and open/close logic:
\[
\sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]

Where all parameters (\( f_i, K_i, d_j, c_{ij} \)) are as listed above.

This model minimizes the total system cost (fixed facility costs plus variable shipping costs) while ensuring all demand is met and no factory exceeds its capacity. The optimal solution will specify which factories to open and the shipment plan from open factories to each distribution center.