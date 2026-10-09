Let:
- $I$ = set of warehouses: $\{ \text{W1}, \text{W2}, \text{W3}, \text{W4}, \text{W5}, \text{W6}, \text{W7}, \text{W8}, \text{W9}, \text{W10} \}$
- $J$ = set of customers: $\{ \text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}, \text{C13}, \text{C14}, \text{C15}, \text{C16}, \text{C17}, \text{C18}, \text{C19}, \text{C20} \}$

Parameters:
- $f_i$ = fixed cost of opening warehouse $i$
- $s_i$ = capacity of warehouse $i$
- $d_j$ = demand of customer $j$
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to customer $j$

Decision variables:
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise
- $x_{ij} \geq 0$: amount supplied from warehouse $i$ to customer $j$

Data (from retrieved files):

Warehouse fixed costs and capacities:
\[
\begin{array}{lll}
\text{Warehouse} & f_i & s_i \\
\text{W1} & 2000 & 1000 \\
\text{W2} & 2500 & 1500 \\
\text{W3} & 1800 & 1200 \\
\text{W4} & 3200 & 2000 \\
\text{W5} & 1500 & 800 \\
\text{W6} & 4000 & 2500 \\
\text{W7} & 2800 & 1800 \\
\text{W8} & 1950 & 1100 \\
\text{W9} & 3500 & 2100 \\
\text{W10} & 2200 & 1300 \\
\end{array}
\]

Customer demands:
\[
\begin{array}{ll}
\text{Customer} & d_j \\
\text{C1} & 800 \\
\text{C2} & 600 \\
\text{C3} & 500 \\
\text{C4} & 700 \\
\text{C5} & 450 \\
\text{C6} & 950 \\
\text{C7} & 350 \\
\text{C8} & 850 \\
\text{C9} & 400 \\
\text{C10} & 750 \\
\text{C11} & 900 \\
\text{C12} & 550 \\
\text{C13} & 650 \\
\text{C14} & 820 \\
\text{C15} & 480 \\
\text{C16} & 920 \\
\text{C17} & 320 \\
\text{C18} & 780 \\
\text{C19} & 520 \\
\text{C20} & 680 \\
\end{array}
\]

Transportation costs $c_{ij}$ (from warehouse $i$ to customer $j$):

|        | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|--------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W1     | 10 | 15 | 20 | 11 | 16 | 18 | 7  | 12 | 22 | 9   | 14  | 19  | 25  | 13  | 17  | 6   | 21  | 15  | 8   | 10  |
| W2     | 18 | 12 | 9  | 14 | 10 | 5  | 19 | 23 | 11 | 16  | 20  | 8   | 15  | 22  | 7   | 13  | 24  | 17  | 12  | 6   |
| W3     | 13 | 17 | 15 | 8  | 12 | 21 | 16 | 10 | 5  | 24  | 13  | 22  | 7   | 19  | 14  | 18  | 9   | 25  | 11  | 16  |
| W4     | 7  | 22 | 11 | 16 | 20 | 8  | 15 | 19 | 13 | 25  | 6   | 14  | 21  | 9   | 23  | 17  | 10  | 18  | 24  | 5   |
| W5     | 16 | 9  | 25 | 13 | 7  | 10 | 23 | 14 | 18 | 21  | 5   | 17  | 9   | 24  | 12  | 20  | 6   | 15  | 19  | 11  |
| W6     | 22 | 6  | 14 | 19 | 23 | 11 | 8  | 17 | 9  | 12  | 15  | 24  | 5   | 20  | 10  | 25  | 13  | 7   | 18  | 16  |
| W7     | 8  | 25 | 17 | 9  | 14 | 22 | 11 | 6  | 16 | 20  | 18  | 13  | 24  | 5   | 19  | 12  | 23  | 10  | 7   | 15  |
| W8     | 19 | 11 | 7  | 21 | 15 | 24 | 13 | 16 | 20 | 8   | 17  | 10  | 12  | 23  | 5   | 14  | 22  | 9   | 16  | 25  |
| W9     | 12 | 20 | 5  | 23 | 17 | 14 | 9  | 25 | 18 | 11  | 16  | 21  | 10  | 7   | 24  | 15  | 19  | 6   | 13  | 22  |
| W10    | 25 | 14 | 22 | 5  | 19 | 12 | 24 | 7  | 15 | 17  | 23  | 6   | 16  | 10  | 20  | 9   | 18  | 11  | 25  | 14  |

Model:

Minimize total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Customer demand satisfaction:
\[
\sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
\]

2. Warehouse capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i y_i \qquad \forall i \in I
\]

3. Variable domains:
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]
\[
y_i \in \{0,1\} \qquad \forall i \in I
\]

Where all parameters and indices are as defined above, and all coefficients and identifiers are as retrieved.