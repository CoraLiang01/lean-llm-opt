##### Decision Variables

For each supplier $i \in I$ and customer $j \in J$,
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

3. **Non-negativity:** For all $i \in I$, $j \in J$,
$$
x_{ij} \geq 0
$$

##### Index Sets and Data Mapping

- $I$: Set of suppliers (stores), from `supply_capacity.csv` and `transportation_costs.csv` row IDs:
  $$
  I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}
  $$
- $J$: Set of customers, from `customer_demand.csv` and `transportation_costs.csv` column IDs:
  $$
  J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}
  $$

- $d_j$: Demand for customer $j$, from `customer_demand.csv`, column `demand`, indexed by `customer`.
- $s_i$: Supply capacity for supplier $i$, from `supply_capacity.csv`, column `supply_capacity`, indexed by `Unnamed: 0`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv`, row `Unnamed: 0`, column $j$.

##### Data Mapping

- $d_j$: `customer_demand.csv`, table_id: `file_0_view_0`, columns: `customer`, `demand`
- $s_i$: `supply_capacity.csv`, table_id: `file_1_view_0`, columns: `Unnamed: 0`, `supply_capacity`
- $c_{ij}$: `transportation_costs.csv`, table_id: `file_2_view_0`, row index: `Unnamed: 0`, column index: $j$ (customer IDs)

##### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
\end{align*}
$$

with all parameters and index sets as mapped above.