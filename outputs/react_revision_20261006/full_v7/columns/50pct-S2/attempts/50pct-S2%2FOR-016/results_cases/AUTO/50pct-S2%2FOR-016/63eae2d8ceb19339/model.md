##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers), as listed in column "supplier_id" of supply_capacity.csv and transportation_costs.csv.
- $J$: set of customer groups, as listed in column "customer_id" of customer_demand.csv and as suffixes in transportation_costs.csv columns.

##### Parameters

- $d_j$: demand (units) for customer group $j \in J$, from "demand_units" in customer_demand.csv.
- $s_i$: supply capacity (units) for distribution center $i \in I$, from "supply_capacity_units" in supply_capacity.csv.
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_cost_to_Ck" columns in transportation_costs.csv.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each distribution center $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

---

##### Data Mapping

- $I$ = all "supplier_id" values in supply_capacity.csv and transportation_costs.csv (S1, S2, ..., S18).
- $J$ = all "customer_id" values in customer_demand.csv and as suffixes in transportation_costs.csv (C1, C2, ..., C18).
- $d_j$ = "demand_units" for customer $j$ in customer_demand.csv, table_id: file_0_view_0, column: demand_units.
- $s_i$ = "supply_capacity_units" for supplier $i$ in supply_capacity.csv, table_id: file_1_view_0, column: supply_capacity_units.
- $c_{ij}$ = "transportation_cost_to_Ck" for supplier $i$ and customer $j$ in transportation_costs.csv, table_id: file_2_view_0, columns: transportation_cost_to_C1 ... transportation_cost_to_C18.

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files, preserving all identifiers and source order.