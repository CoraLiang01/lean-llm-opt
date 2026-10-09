##### Sets

- Warehouses (suppliers): $I = \{S1, S2, S3, S4, S5\}$
- Stores (customers): $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Demand for each store $j$ ($d_j$):

| Store $j$ | $d_j$ |
|-----------|-------|
| D1        | 428   |
| D2        | 217   |
| D3        | 214   |
| D4        | 380   |
| D5        | 254   |

- Supply capacity for each warehouse $i$ ($s_i$):

| Warehouse $i$ | $s_i$ |
|---------------|-------|
| S1            | 428   |
| S2            | 217   |
| S3            | 214   |
| S4            | 380   |
| S5            | 254   |

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

|        | D1           | D2           | D3           | D4           | D5           |
|--------|--------------|--------------|--------------|--------------|--------------|
| S1     | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
| S2     | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.1756820  |
| S3     | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.8493396  |
| S4     | 270.5749848  | 32.50253586  | 4.684209810  | 1.568226969  | 9.589275990  |
| S5     | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.6501532  |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous).

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

2. **Supply capacity:** Each warehouse cannot ship more than its supply capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Parameter Tables

**Store Demands ($d_j$):**

| customer_id | demand_units |
|-------------|-------------|
| D1          | 428         |
| D2          | 217         |
| D3          | 214         |
| D4          | 380         |
| D5          | 254         |

**Warehouse Supply Capacities ($s_i$):**

| supplier_id | supply_capacity_units |
|-------------|---------------------|
| S1          | 428                 |
| S2          | 217                 |
| S3          | 214                 |
| S4          | 380                 |
| S5          | 254                 |

**Transportation Costs ($c_{ij}$):**

| supplier_id | to_D1        | to_D2        | to_D3        | to_D4        | to_D5        |
|-------------|--------------|--------------|--------------|--------------|--------------|
| S1          | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
| S2          | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.1756820  |
| S3          | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.8493396  |
| S4          | 270.5749848  | 32.50253586  | 4.684209810  | 1.568226969  | 9.589275990  |
| S5          | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.6501532  |