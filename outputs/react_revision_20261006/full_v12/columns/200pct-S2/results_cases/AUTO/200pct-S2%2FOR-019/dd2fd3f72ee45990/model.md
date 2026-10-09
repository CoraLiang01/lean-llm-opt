#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from column "supplier_id" in supply_capacity.csv and transportation_costs.csv, and $J$ the set of customer groups from column "customer_id" in customer_demand.csv and transportation_costs.csv.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv).

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand for customer $j$ (from customer_demand.csv).

2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of supplier $i$ (from supply_capacity.csv).

3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$: All "supplier_id" in supply_capacity.csv (file_1_view_0, column "supplier_id") and transportation_costs.csv (file_2_view_0, column "supplier_id").
- $J$: All "customer_id" in customer_demand.csv (file_0_view_0, column "customer_id") and as suffixes in transportation_costs.csv columns "transportation_cost_to_*".
- $d_j$: Demand for customer $j$ from customer_demand.csv (file_0_view_0, columns "customer_id", "demand").
- $s_i$: Supply capacity for supplier $i$ from supply_capacity.csv (file_1_view_0, columns "supplier_id", "supply_capacity").
- $c_{ij}$: Transportation cost from supplier $i$ to customer $j$ from transportation_costs.csv (file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}").

Index sets and all coefficients are defined by the current CSV records and columns as described above. No data is omitted or aggregated.