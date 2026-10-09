#### Decision Variables

For all $i \in I$ (distribution centers), $j \in J$ (customer groups):

$$
x_{ij} \geq 0
$$

= quantity shipped from distribution center $i$ to customer group $j$.

#### Objective Function

$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):

   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$

2. **Supply capacity** (each distribution center does not exceed its capacity):

   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$

3. **Non-negativity**:

   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Index sets:**
  - $I$ (distribution centers): S1, S2, ..., S18 (from "supply_capacity.csv" Unnamed: 0, source order)
  - $J$ (customer groups): C1, C2, ..., C18 (from "customer_demand.csv" customer, source order)

- **Parameters:**
  - $d_j$ (demand for customer $j$): from "customer_demand.csv", column "demand", table_id: file_0_view_0
  - $s_i$ (supply capacity for supplier $i$): from "supply_capacity.csv", column "supply_capacity", table_id: file_1_view_0
  - $c_{ij}$ (cost per unit from $i$ to $j$): from "transportation_costs.csv", table_id: file_2_view_0, row index $i$ (Unnamed: 0), column $j$ (C1...C18)

- **Variable:**
  - $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$

---

#### Explicit Data Mapping

- $I$ (distribution centers): from "supply_capacity.csv", table_id: file_1_view_0, column "Unnamed: 0"
- $J$ (customer groups): from "customer_demand.csv", table_id: file_0_view_0, column "customer"
- $d_j$: from "customer_demand.csv", table_id: file_0_view_0, column "demand"
- $s_i$: from "supply_capacity.csv", table_id: file_1_view_0, column "supply_capacity"
- $c_{ij}$: from "transportation_costs.csv", table_id: file_2_view_0, row "Unnamed: 0" = $i$, column $j$

No data is omitted or aggregated; all identifiers and coefficients are preserved in source order.