##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I$ where $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (distribution centers, from `supply_capacity.csv`, column `Unnamed: 0`, table_id: `file_1_view_0`)
- $j \in J$ where $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (customer groups, from `customer_demand.csv`, column `customer`, table_id: `file_0_view_0`)

##### Parameters

- $d_j$: demand of customer group $j$ (from `customer_demand.csv`, column `demand`, table_id: `file_0_view_0`)
- $s_i$: supply capacity of distribution center $i$ (from `supply_capacity.csv`, column `supply_capacity`, table_id: `file_1_view_0`)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from `transportation_costs.csv`, entry at row $i$ and column $j$, table_id: `file_2_view_0`)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   (Every customer group receives at least its demand.)

2. **Supply capacity:**  
   For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   (No distribution center ships more than its capacity.)

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (distribution centers): all values in `supply_capacity.csv`, column `Unnamed: 0`, table_id: `file_1_view_0`
- $J$ (customer groups): all values in `customer_demand.csv`, column `customer`, table_id: `file_0_view_0`
- $d_j$: for each $j$, value in `customer_demand.csv`, column `demand`, table_id: `file_0_view_0`
- $s_i$: for each $i$, value in `supply_capacity.csv`, column `supply_capacity`, table_id: `file_1_view_0`
- $c_{ij}$: for each $i,j$, value in `transportation_costs.csv`, row $i$ (`Unnamed: 0`), column $j$, table_id: `file_2_view_0`

---

**Index sets, parameters, and all coefficients are bound exactly to the retrieved data and identifiers.**