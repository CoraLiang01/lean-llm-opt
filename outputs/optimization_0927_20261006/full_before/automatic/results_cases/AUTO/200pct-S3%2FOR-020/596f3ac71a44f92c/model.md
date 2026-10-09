##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses) and $j \in J$ (stores).

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (warehouses, in source order)
- $J = \{D1, D2, D3, D4, D5\}$ (stores, in source order)

- Store demands (from customer_demand.csv):
  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Warehouse supply capacities (from supply_capacity.csv):
  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation costs per unit (from transportation_costs.csv):

\[
\begin{array}{c|ccccc}
 & D1 & D2 & D3 & D4 & D5 \\
\hline
S1 & 269.3910588020795 & 1.453733539093394 & 99.60345345756603 & 26.64078166309837 & 9.537688956880922 \\
S2 & 9.291846876785185 & 10.874778437070225 & 144.52609291614627 & 11.420133077898234 & 153.1756819927813 \\
S3 & 9.674584301671008 & 2.6191650959687944 & 100.8242249168735 & 3.212191088791688 & 133.8493396124168 \\
S4 & 270.57498480010247 & 32.50253586 & 4.6842098096469815 & 1.5682269686546804 & 9.58927599 \\
S5 & 226.0331910675782 & 8.669161980826471 & 65.47681316968448 & 9.068765258459958 & 202.65015316425533 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ are the transportation costs as given above.

##### Constraints

1. **Demand satisfaction:** For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
   - $\sum_{i \in I} x_{i,D1} \geq 428$
   - $\sum_{i \in I} x_{i,D2} \geq 217$
   - $\sum_{i \in I} x_{i,D3} \geq 214$
   - $\sum_{i \in I} x_{i,D4} \geq 380$
   - $\sum_{i \in I} x_{i,D5} \geq 254$

2. **Supply capacity:** For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
   - $\sum_{j \in J} x_{S1,j} \leq 428$
   - $\sum_{j \in J} x_{S2,j} \leq 217$
   - $\sum_{j \in J} x_{S3,j} \leq 214$
   - $\sum_{j \in J} x_{S4,j} \leq 380$
   - $\sum_{j \in J} x_{S5,j} \leq 254$

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
with all parameters and coefficients as listed above.