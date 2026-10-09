#### Mathematical Model

Let $I$ be the set of suppliers from column "Unnamed: 0" in supply_capacity.csv and transportation_costs.csv, and $J$ the set of customers from column "customer" in customer_demand.csv and columns "C1"–"C10" in transportation_costs.csv.

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
   where $d_j$ is the demand of customer $j$ (from customer_demand.csv).
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

- $I$: All supplier IDs from column "Unnamed: 0" in supply_capacity.csv and transportation_costs.csv (source: file_1_view_0, file_2_view_0).
- $J$: All customer IDs from column "customer" in customer_demand.csv and columns "C1"–"C10" in transportation_costs.csv (source: file_0_view_0, file_2_view_0).
- $d_j$: Demand for customer $j$ from column "demand" in customer_demand.csv (source: file_0_view_0).
- $s_i$: Supply capacity for supplier $i$ from column "supply_capacity" in supply_capacity.csv (source: file_1_view_0).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from the matrix in transportation_costs.csv, with rows indexed by "Unnamed: 0" (supplier) and columns by customer IDs (source: file_2_view_0).
- $x_{ij}$: Decision variable, continuous and nonnegative, for all $i \in I$, $j \in J$.

All index sets, parameters, and coefficients are mapped directly from the current CSV sources as described above.