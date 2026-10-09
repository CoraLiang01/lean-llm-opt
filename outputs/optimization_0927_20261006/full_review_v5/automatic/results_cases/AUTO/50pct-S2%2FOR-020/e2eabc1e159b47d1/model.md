##### Sets

- Warehouses (suppliers): $I = \{S1, S2, S3, S4, S5\}$
- Stores (customers): $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Demand at store $j$ ($d_j$):

| Store | $d_j$ |
|-------|-------|
| D1    | 428   |
| D2    | 217   |
| D3    | 214   |
| D4    | 380   |
| D5    | 254   |

- Supply capacity at warehouse $i$ ($s_i$):

| Warehouse | $s_i$ |
|-----------|-------|
| S1        | 428   |
| S2        | 217   |
| S3        | 214   |
| S4        | 380   |
| S5        | 254   |

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

|         | D1           | D2           | D3           | D4           | D5           |
|---------|--------------|--------------|--------------|--------------|--------------|
| S1      | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
| S2      | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
| S3      | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.84933961 |
| S4      | 270.5749848  | 32.50253586  | 4.684209810  | 1.568226969  | 9.58927599   |
| S5      | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.65015316 |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous).

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
   Explicitly:
   - $\sum_{i \in I} x_{i,D1} \geq 428$
   - $\sum_{i \in I} x_{i,D2} \geq 217$
   - $\sum_{i \in I} x_{i,D3} \geq 214$
   - $\sum_{i \in I} x_{i,D4} \geq 380$
   - $\sum_{i \in I} x_{i,D5} \geq 254$

2. **Supply capacity:** Each warehouse cannot ship more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
   Explicitly:
   - $\sum_{j \in J} x_{S1,j} \leq 428$
   - $\sum_{j \in J} x_{S2,j} \leq 217$
   - $\sum_{j \in J} x_{S3,j} \leq 214$
   - $\sum_{j \in J} x_{S4,j} \leq 380$
   - $\sum_{j \in J} x_{S5,j} \leq 254$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Parameter Tables

**Demand ($d_j$):**

| customer_id | demand_units |
|-------------|-------------|
| D1          | 428         |
| D2          | 217         |
| D3          | 214         |
| D4          | 380         |
| D5          | 254         |

**Supply Capacity ($s_i$):**

| supplier_id | supply_capacity_units |
|-------------|----------------------|
| S1          | 428                  |
| S2          | 217                  |
| S3          | 214                  |
| S4          | 380                  |
| S5          | 254                  |

**Transportation Costs ($c_{ij}$):**

| supplier_id | to_D1        | to_D2        | to_D3        | to_D4        | to_D5        |
|-------------|--------------|--------------|--------------|--------------|--------------|
| S1          | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
| S2          | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
| S3          | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.84933961 |
| S4          | 270.5749848  | 32.50253586  | 4.684209810  | 1.568226969  | 9.58927599   |
| S5          | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.65015316 |

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

where all sets, parameters, and coefficients are as listed above.