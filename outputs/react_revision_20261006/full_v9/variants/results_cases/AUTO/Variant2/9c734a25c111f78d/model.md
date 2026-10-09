## Symbolic Mathematical Model

**Sets:**
- $I$: set of suppliers (from supplier_capacity.csv), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$
- $J$: set of customers (from customer_demand.csv), $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$

**Parameters:**
- $s_i$: supply capacity of supplier $i \in I$ (from supplier_capacity.csv, column SupplyCapacity, table_id file_0_view_0)
- $d_j$: demand of customer $j \in J$ (from customer_demand.csv, column Demand, table_id file_1_view_0)
- $c_{ij}$: per-unit transportation cost from supplier $i$ to customer $j$ (from route_variable_costs.csv, table_id file_2_view_0, entry at row $i$, column $j$)
- $f_{ij}$: fixed route activation cost for shipping from $i$ to $j$ (from route_fixed_costs.csv, table_id file_3_view_0, entry at row $i$, column $j$)
- $M_{ij}$: a sufficiently large constant for each $(i,j)$, e.g., $M_{ij} = \min(s_i, \sum_{j'} d_{j'})$

**Decision Variables:**
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous)
- $y_{ij} \in \{0,1\}$: 1 if any positive amount is shipped from $i$ to $j$, 0 otherwise (binary)

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} \left( c_{ij} x_{ij} + f_{ij} y_{ij} \right)
\]

**Subject to:**

1. **Supply capacity at each supplier:**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

2. **Demand satisfaction at each customer:**
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   \]

3. **Route activation linking:**
   \[
   x_{ij} \leq M_{ij} y_{ij} \qquad \forall i \in I,\, j \in J
   \]

4. **Nonnegativity and integrality:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- **Suppliers $I$:** All values in column Supplier of supplier_capacity.csv (table_id file_0_view_0)
- **Customers $J$:** All values in column Customer of customer_demand.csv (table_id file_1_view_0)
- **$s_i$:** SupplyCapacity for supplier $i$ in supplier_capacity.csv (file_0_view_0)
- **$d_j$:** Demand for customer $j$ in customer_demand.csv (file_1_view_0)
- **$c_{ij}$:** Entry at row Supplier $i$, column $j$ in route_variable_costs.csv (file_2_view_0)
- **$f_{ij}$:** Entry at row Supplier $i$, column $j$ in route_fixed_costs.csv (file_3_view_0)
- **$x_{ij}$, $y_{ij}$:** Decision variables as defined above, for all $(i,j) \in I \times J$

- **$M_{ij}$:** For each $(i,j)$, a sufficiently large constant, e.g., $M_{ij} = s_i$ or $M_{ij} = \sum_{j'} d_{j'}$ (not from data, but required for linking constraint).

---

**All sets, parameters, and variables are defined exactly as in the source data. No bounds are conditional on activation except for the linking constraint $x_{ij} \leq M_{ij} y_{ij}$.**