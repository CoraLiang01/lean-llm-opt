##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of product shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses) and $j \in J$ (stores).

##### Sets

- Warehouses $I = \{S1, S2, S3, S4, S5\}$
- Stores $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

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

##### Mathematical Model

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each store:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each warehouse:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Where:

- $I = \{S1, S2, S3, S4, S5\}$
- $J = \{D1, D2, D3, D4, D5\}$
- $d_j$ and $s_i$ as listed above
- $c_{ij}$ as in the table above

All variables $x_{ij}$ are continuous and nonnegative.