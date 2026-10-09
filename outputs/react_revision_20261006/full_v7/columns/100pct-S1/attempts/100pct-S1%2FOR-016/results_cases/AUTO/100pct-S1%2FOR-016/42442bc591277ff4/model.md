##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers), as listed in the "supplier_id" column of "supply_capacity.csv" and "transportation_costs.csv".
- $J$: set of customer groups, as listed in the "customer_id" column of "customer_demand.csv" and as columns in "transportation_costs.csv".

##### Parameters

- $d_j$: demand (units) for customer group $j \in J$, from "customer_demand.csv".
- $s_i$: supply capacity (units) for distribution center $i \in I$, from "supply_capacity.csv".
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_costs.csv".

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each customer group must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$

2. **Supply capacity:** Each distribution center cannot ship more than its supply capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ = all "supplier_id" values in "supply_capacity.csv" and "transportation_costs.csv" (source: file_1_view_0, file_2_view_0)
- $J$ = all "customer_id" values in "customer_demand.csv" and as columns in "transportation_costs.csv" (source: file_0_view_0, file_2_view_0)
- $d_j$ = "demand_units" for customer $j$ in "customer_demand.csv" (source: file_0_view_0, columns: customer_id, demand_units)
- $s_i$ = "supply_capacity_units" for supplier $i$ in "supply_capacity.csv" (source: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$ = "transportation_cost_to_Ck" for supplier $i$ and customer $j$ in "transportation_costs.csv" (source: file_2_view_0, row: supplier_id, columns: transportation_cost_to_Ck for each $j$)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files, preserving all identifiers and source order.