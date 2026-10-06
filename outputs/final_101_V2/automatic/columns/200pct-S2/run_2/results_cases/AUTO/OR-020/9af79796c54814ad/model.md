##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of products shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses) and $j \in J$ (stores).

##### Sets

- Warehouses $I = \{S1, S2, S3, S4, S5\}$
- Stores $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Demand for each store (from customer_demand.csv):

  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Supply capacity for each warehouse (from supply_capacity.csv):

  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation costs per unit (from transportation_costs.csv):

  - For $i = S1$:
    - $c_{S1,D1} = 269.3910588020795$
    - $c_{S1,D2} = 1.453733539093394$
    - $c_{S1,D3} = 99.60345345756603$
    - $c_{S1,D4} = 26.64078166309837$
    - $c_{S1,D5} = 9.537688956880922$
  - For $i = S2$:
    - $c_{S2,D1} = 9.291846876785185$
    - $c_{S2,D2} = 10.874778437070225$
    - $c_{S2,D3} = 144.52609291614627$
    - $c_{S2,D4} = 11.420133077898234$
    - $c_{S2,D5} = 153.1756819927813$
  - For $i = S3$:
    - $c_{S3,D1} = 9.674584301671008$
    - $c_{S3,D2} = 2.6191650959687944$
    - $c_{S3,D3} = 100.8242249168735$
    - $c_{S3,D4} = 3.212191088791688$
    - $c_{S3,D5} = 133.8493396124168$
  - For $i = S4$:
    - $c_{S4,D1} = 270.57498480010247$
    - $c_{S4,D2} = 32.50253586$
    - $c_{S4,D3} = 4.6842098096469815$
    - $c_{S4,D4} = 1.5682269686546804$
    - $c_{S4,D5} = 9.58927599$
  - For $i = S5$:
    - $c_{S5,D1} = 226.0331910675782$
    - $c_{S5,D2} = 8.669161980826471$
    - $c_{S5,D3} = 65.47681316968448$
    - $c_{S5,D4} = 9.068765258459958$
    - $c_{S5,D5} = 202.65015316425533$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction (each store receives at least its demand):**

   For all $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

   Explicitly:
   - $\sum_{i \in I} x_{i,D1} \geq 428$
   - $\sum_{i \in I} x_{i,D2} \geq 217$
   - $\sum_{i \in I} x_{i,D3} \geq 214$
   - $\sum_{i \in I} x_{i,D4} \geq 380$
   - $\sum_{i \in I} x_{i,D5} \geq 254$

2. **Supply capacity (each warehouse ships no more than its capacity):**

   For all $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

   Explicitly:
   - $\sum_{j \in J} x_{S1,j} \leq 428$
   - $\sum_{j \in J} x_{S2,j} \leq 217$
   - $\sum_{j \in J} x_{S3,j} \leq 214$
   - $\sum_{j \in J} x_{S4,j} \leq 380$
   - $\sum_{j \in J} x_{S5,j} \leq 254$

3. **Non-negativity:**

   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{i,D1} \geq 428 \\
& \sum_{i \in I} x_{i,D2} \geq 217 \\
& \sum_{i \in I} x_{i,D3} \geq 214 \\
& \sum_{i \in I} x_{i,D4} \geq 380 \\
& \sum_{i \in I} x_{i,D5} \geq 254 \\
& \sum_{j \in J} x_{S1,j} \leq 428 \\
& \sum_{j \in J} x_{S2,j} \leq 217 \\
& \sum_{j \in J} x_{S3,j} \leq 214 \\
& \sum_{j \in J} x_{S4,j} \leq 380 \\
& \sum_{j \in J} x_{S5,j} \leq 254 \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]

where all $c_{ij}$, $d_j$, and $s_i$ are as listed above.