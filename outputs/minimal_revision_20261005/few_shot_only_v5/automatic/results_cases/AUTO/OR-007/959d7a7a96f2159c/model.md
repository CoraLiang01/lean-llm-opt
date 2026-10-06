##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses) and $j \in J$ (stores).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- Demand $d_j$ for each store $j$:
  - $d_{\text{D1}} = 428$
  - $d_{\text{D2}} = 217$
  - $d_{\text{D3}} = 214$
  - $d_{\text{D4}} = 380$
  - $d_{\text{D5}} = 254$
- Supply capacity $s_i$ for each warehouse $i$:
  - $s_{\text{S1}} = 428$
  - $s_{\text{S2}} = 217$
  - $s_{\text{S3}} = 214$
  - $s_{\text{S4}} = 380$
  - $s_{\text{S5}} = 254$
- Transportation cost $c_{ij}$ per unit from warehouse $i$ to store $j$:

|        | D1              | D2              | D3              | D4              | D5              |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1     | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2     | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3     | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4     | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5     | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
   That is,
   - $\sum_{i} x_{i,\text{D1}} \geq 428$
   - $\sum_{i} x_{i,\text{D2}} \geq 217$
   - $\sum_{i} x_{i,\text{D3}} \geq 214$
   - $\sum_{i} x_{i,\text{D4}} \geq 380$
   - $\sum_{i} x_{i,\text{D5}} \geq 254$

2. **Supply capacity:** Each warehouse cannot ship more than its capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
   That is,
   - $\sum_{j} x_{\text{S1},j} \leq 428$
   - $\sum_{j} x_{\text{S2},j} \leq 217$
   - $\sum_{j} x_{\text{S3},j} \leq 214$
   - $\sum_{j} x_{\text{S4},j} \leq 380$
   - $\sum_{j} x_{\text{S5},j} \leq 254$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- customer_demand.csv: $d_j$ for $j \in J$ (store demands)
- supply_capacity.csv: $s_i$ for $i \in I$ (warehouse capacities)
- transportation_costs.csv: $c_{ij}$ for $i \in I$, $j \in J$ (cost matrix, rows = warehouses, columns = stores, source order preserved)