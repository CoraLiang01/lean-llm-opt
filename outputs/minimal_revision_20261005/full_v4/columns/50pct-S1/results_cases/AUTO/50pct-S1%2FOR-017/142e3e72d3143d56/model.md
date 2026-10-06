##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each supplier ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (Suppliers): All `supplier_id` in `supply_capacity.csv` (`file_1_view_0`)
- $J$ (Customers): All `customer_id` in `customer_demand.csv` (`file_0_view_0`)
- $d_j$: Demand for customer $j$ from `demand` in `customer_demand.csv` (`file_0_view_0`)
- $s_i$: Supply capacity for supplier $i$ from `supply_capacity` in `supply_capacity.csv` (`file_1_view_0`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from `transportation_cost_to_Ck` in `transportation_costs.csv` (`file_2_view_0`), where $k$ matches $j$.

##### Table and Column Reference

- **Customers**:  
  Table: `file_0_view_0` (`customer_demand.csv`)  
  Columns: `customer_id`, `demand`
- **Suppliers**:  
  Table: `file_1_view_0` (`supply_capacity.csv`)  
  Columns: `supplier_id`, `supply_capacity`
- **Transportation Costs**:  
  Table: `file_2_view_0` (`transportation_costs.csv`)  
  Columns: `supplier_id`, `transportation_cost_to_C1`, ..., `transportation_cost_to_C10`  
  (Column `transportation_cost_to_Ck` gives $c_{ik}$ for supplier $i$ and customer $k$.)

##### Index Sets

- $I = \{$ all `supplier_id` in `file_1_view_0` $\}$
- $J = \{$ all `customer_id` in `file_0_view_0` $\}$

##### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
$$

where all parameters and index sets are as mapped above.