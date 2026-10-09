##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined in the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand of customer $j$.

2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of supplier $i$.

3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$: supplier IDs from column "supplier_id" in table_id "file_1_view_0" (supply_capacity.csv) and "file_2_view_0" (transportation_costs.csv)
- $J$: customer IDs from column "customer_id" in table_id "file_0_view_0" (customer_demand.csv) and as suffixes in "transportation_cost_to_C*" columns in "file_2_view_0"
- $d_j$: demand for customer $j$ from column "demand" in table_id "file_0_view_0"
- $s_i$: supply capacity for supplier $i$ from column "supply_capacity" in table_id "file_1_view_0"
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ from column "transportation_cost_to_Cj" in table_id "file_2_view_0", row "supplier_id" = $i$, column suffix $j$

Index sets $I$ and $J$ are defined by all supplier and customer IDs present in the current source data. All constraints and parameters are mapped directly from the corresponding columns and rows as described above.