##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of products shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses) and $j \in J$ (stores).

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (warehouses)
- $J = \{D1, D2, D3, D4, D5\}$ (stores)

- Store demands (units):

  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Warehouse supply capacities (units):

  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation costs per unit ($c_{ij}$):

  |         | D1           | D2           | D3           | D4           | D5           |
  |---------|--------------|--------------|--------------|--------------|--------------|
  | S1      | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
  | S2      | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
  | S3      | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.84933961 |
  | S4      | 270.5749848  | 32.50253586  | 4.684209810  | 1.568226969  | 9.58927599   |
  | S5      | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.65015316 |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction (each store receives at least its demand):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   Specifically:
   - $\sum_{i \in I} x_{i,D1} \geq 428$
   - $\sum_{i \in I} x_{i,D2} \geq 217$
   - $\sum_{i \in I} x_{i,D3} \geq 214$
   - $\sum_{i \in I} x_{i,D4} \geq 380$
   - $\sum_{i \in I} x_{i,D5} \geq 254$

2. **Supply capacity (each warehouse ships no more than its capacity):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   Specifically:
   - $\sum_{j \in J} x_{S1,j} \leq 428$
   - $\sum_{j \in J} x_{S2,j} \leq 217$
   - $\sum_{j \in J} x_{S3,j} \leq 214$
   - $\sum_{j \in J} x_{S4,j} \leq 380$
   - $\sum_{j \in J} x_{S5,j} \leq 254$

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
\]

Where all parameters and coefficients are as listed above.