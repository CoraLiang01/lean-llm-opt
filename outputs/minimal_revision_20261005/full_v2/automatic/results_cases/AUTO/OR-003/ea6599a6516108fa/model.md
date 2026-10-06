##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i$ to customer group $j$.

- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from "customer_demand.csv", column "demand", table_id: file_0_view_0)
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv", column "supply_capacity", table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv", columns $j$, table_id: file_2_view_0, row $i$)

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
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): "Unnamed: 0" column, table_id: file_1_view_0 and file_2_view_0 rows
- $J$ (customers): "customer" column, table_id: file_0_view_0 and file_2_view_0 columns
- $d_j$: "demand" column, table_id: file_0_view_0, indexed by "customer"
- $s_i$: "supply_capacity" column, table_id: file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: entry in table_id: file_2_view_0, row "Unnamed: 0" = $i$, column $j$

---

**All indices, coefficients, and constraints are bound exactly to the retrieved data and identifiers.**