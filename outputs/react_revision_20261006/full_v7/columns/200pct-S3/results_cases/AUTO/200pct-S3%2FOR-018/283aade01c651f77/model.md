##### Decision Variables

For each supplier $i \in I$ and customer $j \in J$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each supplier $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (Suppliers): All supplier_id in `supply_capacity.csv` and `transportation_costs.csv` (S1, S2, ..., S12)
- $J$ (Customers): All customer_id in `customer_demand.csv` and columns in `transportation_costs.csv` (C1, C2, ..., C12)
- $d_j$: Demand for customer $j$ from `customer_demand.csv` (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: Supply capacity for supplier $i$ from `supply_capacity.csv` (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from `transportation_costs.csv` (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck for customer $k$)

Index sets and all coefficients are defined by the full set of records and columns in the retrieved tables. No data is omitted or aggregated.