##### Mathematical Model

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
   where $d_j$ is the demand for customer $j$.

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
- $d_j$ = demand_units for customer_id $j$ in file_0_view_0
- $s_i$ = supply_capacity_units for supplier_id $i$ in file_1_view_0
- $c_{ij}$ = transportation_cost_to_$j$ for supplier_id $i$ in file_2_view_0 (transportation_costs.csv), with $j$ mapped as in the relationships object

##### Data Mapping

- $I$: All supplier_id in table_id file_1_view_0 (supply_capacity.csv)
- $J$: All customer_id in table_id file_0_view_0 (customer_demand.csv)
- $d_j$: Column demand_units in file_0_view_0, indexed by customer_id
- $s_i$: Column supply_capacity_units in file_1_view_0, indexed by supplier_id
- $c_{ij}$: Column transportation_cost_to_Ck in file_2_view_0, where $i$ = supplier_id, $j$ = customer_id $Ck$ (see relationships.column_id_mapping for mapping), indexed by (supplier_id, customer_id)

- Decision variables: $x_{ij} \geq 0$ continuous, for all $i \in I$, $j \in J$

All index sets, parameters, and constraints are defined exactly as in the current data and user query. No data is omitted or aggregated.