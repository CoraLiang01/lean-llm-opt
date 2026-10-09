#### Mathematical Model

Let $I$ be the set of suppliers (from column "supplier_id" in file_1_view_0), and $J$ the set of customer groups (from column "customer_id" in file_0_view_0).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, column "transportation_cost_to_{j}" for supplier $i$).

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand of customer $j$ (from file_0_view_0, column "demand").

2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity").

3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$: All "supplier_id" in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All "customer_id" in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", for customer $j$.
- $s_i$: file_1_view_0, column "supply_capacity", for supplier $i$.
- $c_{ij}$: file_2_view_0, row with "supplier_id" = $i$, column "transportation_cost_to_{j}" (e.g., "transportation_cost_to_C1" for $j$ = C1).

All indices, parameters, and coefficients are to be taken exactly as listed in the current CSV data, preserving source order and identifiers.