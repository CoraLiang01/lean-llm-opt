##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10$\}$ (from `supply_capacity.csv`, column `Unnamed: 0`, table_id: `file_1_view_0`)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10$\}$ (from `customer_demand.csv`, column `customer`, table_id: `file_0_view_0`)

##### Parameters

- $d_j$: demand of customer $j$ (from `customer_demand.csv`, column `demand`, table_id: `file_0_view_0`)
- $s_i$: supply capacity of supplier $i$ (from `supply_capacity.csv`, column `supply_capacity`, table_id: `file_1_view_0`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from `transportation_costs.csv`, columns $J$, rows $I$, table_id: `file_2_view_0`)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): from `supply_capacity.csv`, column `Unnamed: 0`, table_id: `file_1_view_0`
- $J$ (customers): from `customer_demand.csv`, column `customer`, table_id: `file_0_view_0`
- $d_j$: from `customer_demand.csv`, column `demand`, table_id: `file_0_view_0`
- $s_i$: from `supply_capacity.csv`, column `supply_capacity`, table_id: `file_1_view_0`
- $c_{ij}$: from `transportation_costs.csv`, row `Unnamed: 0$=i$`, column `$j$`, table_id: `file_2_view_0`