##### Decision Variables

For each distribution center (supplier) $i \in I$ and customer group $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

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

- $I$ (Suppliers): All unique `supplier_id` in `file_1_view_0` (supply_capacity.csv) and `file_2_view_0` (transportation_costs.csv)  
  $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$

- $J$ (Customers): All unique `customer_id` in `file_0_view_0` (customer_demand.csv) and columns in `file_2_view_0` (transportation_costs.csv)  
  $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

- $d_j$: Demand for customer $j$  
  Source: `file_0_view_0`, column `demand_units`, key `customer_id`  
  $d_j = \text{demand_units}[\text{customer_id}=j]$

- $s_i$: Supply capacity for supplier $i$  
  Source: `file_1_view_0`, column `supply_capacity_units`, key `supplier_id`  
  $s_i = \text{supply_capacity_units}[\text{supplier_id}=i]$

- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$  
  Source: `file_2_view_0`, row `supplier_id` $=i$, column `transportation_cost_to_{j}`  
  $c_{ij} = \text{transportation_cost_to}_{j}[\text{supplier_id}=i]$

---

#### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}, \text{S13}, \text{S14}, \text{S15}, \text{S16}, \text{S17}, \text{S18}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}, \text{C13}, \text{C14}, \text{C15}, \text{C16}, \text{C17}, \text{C18}\}$

---

#### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \quad && \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad && \forall i \in I \\
& x_{ij} \geq 0 \quad && \forall i \in I,\, j \in J
\end{align*}
$$

---

#### Data Source Mapping

- $d_j$: `file_0_view_0` (`customer_demand.csv`), columns: `customer_id`, `demand_units`
- $s_i$: `file_1_view_0` (`supply_capacity.csv`), columns: `supplier_id`, `supply_capacity_units`
- $c_{ij}$: `file_2_view_0` (`transportation_costs.csv`), row: `supplier_id`, column: `transportation_cost_to_{customer_id}`

All index sets, parameters, and constraints are bound exactly to the retrieved data.