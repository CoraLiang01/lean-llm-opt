##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants, from supply_capacity.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from customer_demand.csv, column "customer", table_id: file_0_view_0)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv, column "demand", table_id: file_0_view_0)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv, columns "C1", "C2", "C3", "C4", table_id: file_2_view_0, rows indexed by "Unnamed: 0")

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each retail outlet $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   where $d_j$ is from customer_demand.csv, table_id: file_0_view_0, column "demand".

2. **Supply capacity:**  
   For each plant $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   where $s_i$ is from supply_capacity.csv, table_id: file_1_view_0, column "supply_capacity".

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (plants): file_1_view_0, column "Unnamed: 0"  
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$ (retail outlets): file_0_view_0, column "customer"  
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, columns "C1", "C2", "C3", "C4", rows indexed by "Unnamed: 0" (plants)

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.**