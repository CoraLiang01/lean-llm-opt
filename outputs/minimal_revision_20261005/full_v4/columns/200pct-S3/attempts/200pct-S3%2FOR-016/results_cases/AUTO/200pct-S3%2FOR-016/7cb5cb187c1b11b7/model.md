##### Decision Variables

For each distribution center (supplier) $i \in I$ and customer group $j \in J$:

$x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (Suppliers): All supplier_id in `supply_capacity.csv` (table_id: file_1_view_0)
- $J$ (Customers): All customer_id in `customer_demand.csv` (table_id: file_0_view_0)
- $d_j$: Demand for customer $j$ from `demand_units` in `customer_demand.csv` (table_id: file_0_view_0, column: demand_units, key: customer_id)
- $s_i$: Supply capacity for supplier $i$ from `supply_capacity_units` in `supply_capacity.csv` (table_id: file_1_view_0, column: supply_capacity_units, key: supplier_id)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from `transportation_costs.csv` (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_{customer_id})

---

#### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18$\}$

---

#### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

---

#### Data Table References

- $d_j$: file_0_view_0, columns: customer_id, demand_units
- $s_i$: file_1_view_0, columns: supplier_id, supply_capacity_units
- $c_{ij}$: file_2_view_0, row: supplier_id, column: transportation_cost_to_{customer_id}