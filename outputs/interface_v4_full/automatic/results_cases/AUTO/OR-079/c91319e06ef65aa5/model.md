##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if factory $i$ is constructed, 0 otherwise.

##### Parameters

- $I = \{A1, A2, \ldots, A15\}$ (factories)
- $J = \{B1, B2, \ldots, B8\}$ (distribution centers)

- Fixed costs $f_i$ and capacities $K_i$ for each $i \in I$:

| $i$  | $f_i$ | $K_i$ |
|------|-------|-------|
| A1   | 0     | 30    |
| A2   | 175   | 10    |
| A3   | 300   | 20    |
| A4   | 375   | 30    |
| A5   | 500   | 40    |
| A6   | 200   | 20    |
| A7   | 260   | 25    |
| A8   | 220   | 30    |
| A9   | 320   | 35    |
| A10  | 280   | 20    |
| A11  | 350   | 40    |
| A12  | 420   | 25    |
| A13  | 470   | 30    |
| A14  | 520   | 50    |
| A15  | 560   | 45    |

- Demand $d_j$ for each $j \in J$:

| $j$  | $d_j$ |
|------|-------|
| B1   | 30    |
| B2   | 25    |
| B3   | 20    |
| B4   | 35    |
| B5   | 25    |
| B6   | 30    |
| B7   | 25    |
| B8   | 30    |

- Shipping costs $c_{ij}$ for each $i \in I$, $j \in J$:

|      | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|------|----|----|----|----|----|----|----|----|
| A1   | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
| A2   | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
| A3   | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
| A4   | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
| A5   | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
| A6   | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
| A7   | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
| A8   | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
| A9   | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
| A10  | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
| A11  | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
| A12  | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
| A13  | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
| A14  | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
| A15  | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity (only if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### All required parameters (vectors and matrices):

- $I = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]$
- $J = [B1, B2, B3, B4, B5, B6, B7, B8]$
- $f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560]$
- $K = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45]$
- $d = [30, 25, 20, 35, 25, 30, 25, 30]$
- $C =$
\[
\begin{bmatrix}
8 & 4 & 3 & 6 & 7 & 5 & 9 & 8 \\
5 & 2 & 3 & 5 & 6 & 4 & 7 & 6 \\
4 & 3 & 4 & 6 & 5 & 5 & 6 & 7 \\
9 & 7 & 5 & 8 & 9 & 6 & 10 & 7 \\
10 & 4 & 2 & 6 & 8 & 5 & 7 & 3 \\
6 & 5 & 4 & 5 & 7 & 6 & 8 & 5 \\
7 & 6 & 5 & 4 & 6 & 7 & 9 & 6 \\
5 & 4 & 6 & 3 & 5 & 6 & 7 & 6 \\
8 & 7 & 6 & 7 & 9 & 8 & 10 & 7 \\
6 & 5 & 7 & 4 & 6 & 5 & 7 & 5 \\
9 & 6 & 4 & 6 & 8 & 7 & 9 & 6 \\
7 & 5 & 6 & 5 & 6 & 5 & 8 & 5 \\
8 & 6 & 5 & 6 & 7 & 6 & 8 & 7 \\
9 & 5 & 3 & 5 & 7 & 4 & 6 & 4 \\
10 & 6 & 4 & 5 & 8 & 5 & 7 & 5 \\
\end{bmatrix}
\]
where row $i$ corresponds to $A_i$ and column $j$ to $B_j$.

---

This model determines which factories to construct and how much to ship from each open factory to each distribution center, minimizing total fixed and shipping costs, while meeting all demand and respecting factory capacities. All parameters are explicitly listed as required.