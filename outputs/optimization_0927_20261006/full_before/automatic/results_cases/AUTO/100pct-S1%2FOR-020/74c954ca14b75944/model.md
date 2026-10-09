##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses), $j \in J$ (stores).

##### Sets

- $I = \{S1, S2, S3, S4, S5\}$ (warehouses)
- $J = \{D1, D2, D3, D4, D5\}$ (stores)

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

  - $c_{S1,D1} = 269.3910588020795$
  - $c_{S1,D2} = 1.453733539093394$
  - $c_{S1,D3} = 99.60345345756603$
  - $c_{S1,D4} = 26.64078166309837$
  - $c_{S1,D5} = 9.537688956880922$

  - $c_{S2,D1} = 9.291846876785185$
  - $c_{S2,D2} = 10.874778437070225$
  - $c_{S2,D3} = 144.52609291614627$
  - $c_{S2,D4} = 11.420133077898234$
  - $c_{S2,D5} = 153.1756819927813$

  - $c_{S3,D1} = 9.674584301671008$
  - $c_{S3,D2} = 2.6191650959687944$
  - $c_{S3,D3} = 100.8242249168735$
  - $c_{S3,D4} = 3.212191088791688$
  - $c_{S3,D5} = 133.8493396124168$

  - $c_{S4,D1} = 270.57498480010247$
  - $c_{S4,D2} = 32.50253586$
  - $c_{S4,D3} = 4.6842098096469815$
  - $c_{S4,D4} = 1.5682269686546804$
  - $c_{S4,D5} = 9.58927599$

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

1. **Demand satisfaction (each store must receive at least its demand):**

   For all $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply capacity (each warehouse cannot ship more than its capacity):**

   For all $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity:**

   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

##### Complete Numerical Model

\[
\begin{align*}
\min\quad & 
269.3910588020795\,x_{S1,D1} + 1.453733539093394\,x_{S1,D2} + 99.60345345756603\,x_{S1,D3} + 26.64078166309837\,x_{S1,D4} + 9.537688956880922\,x_{S1,D5} \\
&+ 9.291846876785185\,x_{S2,D1} + 10.874778437070225\,x_{S2,D2} + 144.52609291614627\,x_{S2,D3} + 11.420133077898234\,x_{S2,D4} + 153.1756819927813\,x_{S2,D5} \\
&+ 9.674584301671008\,x_{S3,D1} + 2.6191650959687944\,x_{S3,D2} + 100.8242249168735\,x_{S3,D3} + 3.212191088791688\,x_{S3,D4} + 133.8493396124168\,x_{S3,D5} \\
&+ 270.57498480010247\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.6842098096469815\,x_{S4,D3} + 1.5682269686546804\,x_{S4,D4} + 9.58927599\,x_{S4,D5} \\
&+ 226.0331910675782\,x_{S5,D1} + 8.669161980826471\,x_{S5,D2} + 65.47681316968448\,x_{S5,D3} + 9.068765258459958\,x_{S5,D4} + 202.65015316425533\,x_{S5,D5}
\end{align*}
\]

Subject to:

For each store:
\[
\begin{align*}
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &\geq 428 \\
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &\geq 217 \\
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &\geq 214 \\
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &\geq 380 \\
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &\geq 254 \\
\end{align*}
\]

For each warehouse:
\[
\begin{align*}
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
\end{align*}
\]

And for all $i \in I$, $j \in J$:
\[
x_{ij} \geq 0
\]