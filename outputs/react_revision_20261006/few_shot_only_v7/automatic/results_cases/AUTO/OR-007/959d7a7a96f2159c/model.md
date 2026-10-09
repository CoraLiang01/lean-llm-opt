##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction (for each store):**
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$

2. **Supply capacity (for each warehouse):**
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **customer_demand.csv**  
  - $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$
  - $d_{\text{D1}} = 428$
  - $d_{\text{D2}} = 217$
  - $d_{\text{D3}} = 214$
  - $d_{\text{D4}} = 380$
  - $d_{\text{D5}} = 254$

- **supply_capacity.csv**  
  - $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
  - $s_{\text{S1}} = 428$
  - $s_{\text{S2}} = 217$
  - $s_{\text{S3}} = 214$
  - $s_{\text{S4}} = 380$
  - $s_{\text{S5}} = 254$

- **transportation_costs.csv**  
  - $c_{ij}$: For $i$ in $I$, $j$ in $J$, $c_{ij}$ is the value in row $i$, column $j$.

  |        | D1              | D2              | D3              | D4              | D5              |
  |--------|-----------------|-----------------|-----------------|-----------------|-----------------|
  | S1     | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
  | S2     | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
  | S3     | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
  | S4     | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
  | S5     | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

- $x_{ij}$ are continuous, nonnegative variables for all $i \in I$, $j \in J$.