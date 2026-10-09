##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers), from all supplier_id in supply_capacity.csv and transportation_costs.csv.
- $J$: set of customer groups, from all customer_id in customer_demand.csv and transportation_costs.csv.

##### Parameters

- $d_j$: demand (units) for customer group $j \in J$, from customer_demand.csv.
- $s_i$: supply capacity (units) for distribution center $i \in I$, from supply_capacity.csv.
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from transportation_costs.csv.

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

##### Data Mapping

- $I$ = all supplier_id in supply_capacity.csv and transportation_costs.csv:  
  $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18$\}$
- $J$ = all customer_id in customer_demand.csv and transportation_costs.csv:  
  $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18$\}$
- $d_j$ = demand_units for customer_id $j$ in customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$ = supply_capacity_units for supplier_id $i$ in supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$ = transportation_cost_to_$j$ for supplier_id $i$ in transportation_costs.csv (table_id: file_2_view_0, columns: supplier_id, transportation_cost_to_C1, ..., transportation_cost_to_C18)

All indices, parameters, and coefficients are mapped directly from the retrieved CSV files as described above.