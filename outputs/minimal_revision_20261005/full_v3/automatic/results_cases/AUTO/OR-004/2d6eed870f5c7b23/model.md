##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I$ (distribution centers): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J$ (customer groups): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from "customer_demand.csv", table_id: file_0_view_0, column: demand)
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv", table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv", table_id: file_2_view_0, row_id: $i$, column_id: $j$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each distribution center does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (distribution centers): row_id_mapping from "supply_capacity.csv" and "transportation_costs.csv" (table_id: file_1_view_0 and file_2_view_0, column: Unnamed: 0)
- $J$ (customer groups): column_id_mapping from "customer_demand.csv" and "transportation_costs.csv" (table_id: file_0_view_0 and file_2_view_0, column: customer)
- $d_j$: "customer_demand.csv", table_id: file_0_view_0, column: demand, indexed by customer
- $s_i$: "supply_capacity.csv", table_id: file_1_view_0, column: supply_capacity, indexed by Unnamed: 0
- $c_{ij}$: "transportation_costs.csv", table_id: file_2_view_0, row_id: $i$ (Unnamed: 0), column_id: $j$ (customer group)

---

**All identifiers and coefficients are bound directly to the supplied data and mappings.**