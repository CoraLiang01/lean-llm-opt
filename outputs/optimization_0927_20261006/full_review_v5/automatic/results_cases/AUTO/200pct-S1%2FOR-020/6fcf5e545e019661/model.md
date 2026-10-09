##### Sets

- Warehouses (suppliers): $I = \{S1, S2, S3, S4, S5\}$
- Stores (customers): $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Demand at store $j$ ($d_j$):

| Store | Demand ($d_j$) |
|-------|---------------|
| D1    | 428           |
| D2    | 217           |
| D3    | 214           |
| D4    | 380           |
| D5    | 254           |

- Supply capacity at warehouse $i$ ($s_i$):

| Warehouse | Supply Capacity ($s_i$) |
|-----------|------------------------|
| S1        | 428                    |
| S2        | 217                    |
| S3        | 214                    |
| S4        | 380                    |
| S5        | 254                    |

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

|         | D1           | D2           | D3           | D4           | D5           |
|---------|--------------|--------------|--------------|--------------|--------------|
| S1      | 269.3910588  | 1.4537335    | 99.6034535   | 26.6407817   | 9.53768896   |
| S2      | 9.29184688   | 10.8747784   | 144.5260929  | 11.42013308  | 153.1756820  |
| S3      | 9.67458430   | 2.61916510   | 100.8242249  | 3.21219109   | 133.8493396  |
| S4      | 270.5749848  | 32.50253586  | 4.68420981   | 1.56822697   | 9.58927599   |
| S5      | 226.0331911  | 8.66916198   | 65.47681317  | 9.06876526   | 202.6501532  |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** Each store must receive at least its demand.
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]

2. **Supply capacity:** Each warehouse cannot ship more than its supply capacity.
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
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

Where all parameters and sets are as defined above.