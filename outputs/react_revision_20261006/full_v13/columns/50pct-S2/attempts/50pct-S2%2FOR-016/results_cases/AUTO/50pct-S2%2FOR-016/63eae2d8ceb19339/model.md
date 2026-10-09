#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

**Subject to:**

1. **Demand satisfaction:**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   where $d_j$ is the demand of customer $j$.

2. **Supply capacity:**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   where $s_i$ is the supply capacity of supplier $i$.

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

**Index sets and parameter mapping:**

- $I$ = set of all supplier_id in file_1_view_0 (supply_capacity.csv)
- $J$ = set of all customer_id in file_0_view_0 (customer_demand.csv)
- $d_j$ = demand_units for customer $j$ from file_0_view_0 (customer_demand.csv)
- $s_i$ = supply_capacity_units for supplier $i$ from file_1_view_0 (supply_capacity.csv)
- $c_{ij}$ = transportation_cost_to_$j$ for supplier $i$ from file_2_view_0 (transportation_costs.csv), with column mapping as per the relationships in the Observation

#### Data Mapping

- $I$: All supplier_id in table_id file_1_view_0 (supply_capacity.csv)
- $J$: All customer_id in table_id file_0_view_0 (customer_demand.csv)
- $d_j$: file_0_view_0, column demand_units, key customer_id
- $s_i$: file_1_view_0, column supply_capacity_units, key supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id, column transportation_cost_to_Ck (where Ck = customer_id), as mapped in the Observation relationships

All indices, parameters, and constraints are defined directly from the current source data, preserving all identifiers and coefficients. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source data.