##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I$ (distribution centers): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J$ (customer groups): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from "customer_demand.csv", table_id: file_0_view_0, column: demand)
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv", table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv", table_id: file_2_view_0, columns $j$, row $i$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each distribution center $i$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (distribution centers): all values in column "Unnamed: 0" of table_id: file_1_view_0 and file_2_view_0 (rows aligned)
- $J$ (customer groups): all values in column "customer" of table_id: file_0_view_0 and columns "C1"–"C12" of table_id: file_2_view_0 (columns aligned)
- $d_j$: value in column "demand" for customer $j$ in table_id: file_0_view_0
- $s_i$: value in column "supply_capacity" for supplier $i$ in table_id: file_1_view_0
- $c_{ij}$: value in column $j$ and row $i$ in table_id: file_2_view_0

All indices and coefficients are to be used exactly as retrieved; no aggregation or omission. Variable domains, objective sense, and constraint forms are as specified above.