##### Decision Variables

For each distribution center $i$ (supplier) and customer group $j$ (customer), let
$$
x_{ij} \geq 0
$$
be the continuous quantity shipped from supplier $i$ to customer $j$.

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
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### Index Sets

- $I$ = set of all supplier IDs from `"supply_capacity.csv"` and `"transportation_costs.csv"`:  
  $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$
- $J$ = set of all customer IDs from `"customer_demand.csv"` and `"transportation_costs.csv"`:  
  $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

##### Data Mapping

- $d_j$ = demand for customer $j$ from `"customer_demand.csv"`  
  Table: `file_0_view_0`, columns: `customer_id`, `demand_units`
- $s_i$ = supply capacity for supplier $i$ from `"supply_capacity.csv"`  
  Table: `file_1_view_0`, columns: `supplier_id`, `supply_capacity_units`
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ from `"transportation_costs.csv"`  
  Table: `file_2_view_0`, row: `supplier_id`, column: `transportation_cost_to_{j}`

##### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
$$

where all parameters and index sets are defined exactly as above from the retrieved data.