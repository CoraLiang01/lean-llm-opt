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

|         | D1                | D2                | D3                | D4                | D5                |
|---------|-------------------|-------------------|-------------------|-------------------|-------------------|
| S1      | 269.3910588020795 | 1.453733539093394 | 99.60345345756603 | 26.64078166309837 | 9.537688956880922 |
| S2      | 9.291846876785185 | 10.874778437070225| 144.52609291614627| 11.420133077898234| 153.1756819927813 |
| S3      | 9.674584301671008 | 2.6191650959687944| 100.8242249168735 | 3.212191088791688 | 133.8493396124168 |
| S4      | 270.57498480010247| 32.50253586       | 4.6842098096469815| 1.5682269686546804| 9.58927599        |
| S5      | 226.0331910675782 | 8.669161980826471 | 65.47681316968448 | 9.068765258459958 | 202.65015316425533|

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
   That is,
   - $\sum_{i} x_{i,D1} \geq 428$
   - $\sum_{i} x_{i,D2} \geq 217$
   - $\sum_{i} x_{i,D3} \geq 214$
   - $\sum_{i} x_{i,D4} \geq 380$
   - $\sum_{i} x_{i,D5} \geq 254$

2. **Supply capacity:** Each warehouse cannot ship more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
   That is,
   - $\sum_{j} x_{S1,j} \leq 428$
   - $\sum_{j} x_{S2,j} \leq 217$
   - $\sum_{j} x_{S3,j} \leq 214$
   - $\sum_{j} x_{S4,j} \leq 380$
   - $\sum_{j} x_{S5,j} \leq 254$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Parameter Tables

**Demand:**

| customer_id | demand_units |
|-------------|-------------|
| D1          | 428         |
| D2          | 217         |
| D3          | 214         |
| D4          | 380         |
| D5          | 254         |

**Supply Capacity:**

| supplier_id | supply_capacity_units |
|-------------|----------------------|
| S1          | 428                  |
| S2          | 217                  |
| S3          | 214                  |
| S4          | 380                  |
| S5          | 254                  |

**Transportation Costs:**

| supplier_id | D1                | D2                | D3                | D4                | D5                |
|-------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| S1          | 269.3910588020795 | 1.453733539093394 | 99.60345345756603 | 26.64078166309837 | 9.537688956880922 |
| S2          | 9.291846876785185 | 10.874778437070225| 144.52609291614627| 11.420133077898234| 153.1756819927813 |
| S3          | 9.674584301671008 | 2.6191650959687944| 100.8242249168735 | 3.212191088791688 | 133.8493396124168 |
| S4          | 270.57498480010247| 32.50253586       | 4.6842098096469815| 1.5682269686546804| 9.58927599        |
| S5          | 226.0331910675782 | 8.669161980826471 | 65.47681316968448 | 9.068765258459958 | 202.65015316425533|