##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to retail store $j$, for all $i \in I$ and $j \in J$ (continuous).

##### Sets

- $I = \{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10\}$ (warehouses)
- $J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10\}$ (retail stores)

##### Parameters

- Demand $d_j$ for each store $j$:
  - $d_{C1} = 45$
  - $d_{C2} = 23$
  - $d_{C3} = 94$
  - $d_{C4} = 92$
  - $d_{C5} = 57$
  - $d_{C6} = 52$
  - $d_{C7} = 23$
  - $d_{C8} = 99$
  - $d_{C9} = 99$
  - $d_{C10} = 77$

- Supply capacity $s_i$ for each warehouse $i$:
  - $s_{S1} = 127$
  - $s_{S2} = 236$
  - $s_{S3} = 168$
  - $s_{S4} = 115$
  - $s_{S5} = 280$
  - $s_{S6} = 179$
  - $s_{S7} = 135$
  - $s_{S8} = 263$
  - $s_{S9} = 283$
  - $s_{S10} = 476$

- Transportation cost $c_{ij}$ per unit from warehouse $i$ to store $j$:

|        | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S2     | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S3     | 79.92102960  | 474.24509131 | 1477.0676289 | 22.58309959  | 474.24509131 | 41.10659696  | 474.24509131 | 474.24509131 | 624.16253950 | 474.24509131 |
| S4     | 1659.3369291 | 57.20541469  | 186.15190481 | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.56338707 | 1029.6974644 |
| S5     | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6     | 1998.9090659 | 985.31654357 | 2.85416869   | 1149.5359675 | 985.31654357 | 730.69236477 | 54.73980798  | 985.31654357 | 46.80310221  | 1149.5359675 |
| S7     | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S8     | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.35062391 | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9     | 98.90755583  | 0.0          | 978.03476648 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S10    | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.14023080 | 0.0          |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
   That is,
   \begin{align*}
   \sum_{i \in I} x_{i,C1} &\geq 45 \\
   \sum_{i \in I} x_{i,C2} &\geq 23 \\
   \sum_{i \in I} x_{i,C3} &\geq 94 \\
   \sum_{i \in I} x_{i,C4} &\geq 92 \\
   \sum_{i \in I} x_{i,C5} &\geq 57 \\
   \sum_{i \in I} x_{i,C6} &\geq 52 \\
   \sum_{i \in I} x_{i,C7} &\geq 23 \\
   \sum_{i \in I} x_{i,C8} &\geq 99 \\
   \sum_{i \in I} x_{i,C9} &\geq 99 \\
   \sum_{i \in I} x_{i,C10} &\geq 77 \\
   \end{align*}

2. **Supply capacity:** For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
   That is,
   \begin{align*}
   \sum_{j \in J} x_{S1,j} &\leq 127 \\
   \sum_{j \in J} x_{S2,j} &\leq 236 \\
   \sum_{j \in J} x_{S3,j} &\leq 168 \\
   \sum_{j \in J} x_{S4,j} &\leq 115 \\
   \sum_{j \in J} x_{S5,j} &\leq 280 \\
   \sum_{j \in J} x_{S6,j} &\leq 179 \\
   \sum_{j \in J} x_{S7,j} &\leq 135 \\
   \sum_{j \in J} x_{S8,j} &\leq 263 \\
   \sum_{j \in J} x_{S9,j} &\leq 283 \\
   \sum_{j \in J} x_{S10,j} &\leq 476 \\
   \end{align*}

3. **Non-negativity:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]
with all parameters and indices as specified above.