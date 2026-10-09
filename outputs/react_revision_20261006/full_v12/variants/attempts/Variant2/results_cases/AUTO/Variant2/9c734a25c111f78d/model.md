## Mixed-Integer Fixed-Charge Transportation Model

**Sets:**
- $I$: set of suppliers (from supplier_capacity.csv), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$
- $J$: set of customers (from customer_demand.csv), $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$

**Parameters:**
- $a_i$: supply capacity of supplier $i$ (from supplier_capacity.csv, column SupplyCapacity, table_id file_0_view_0)
- $b_j$: demand of customer $j$ (from customer_demand.csv, column Demand, table_id file_1_view_0)
- $c_{ij}$: variable transportation cost per unit from supplier $i$ to customer $j$ (from route_variable_costs.csv, table_id file_2_view_0, entry [i,j])
- $f_{ij}$: fixed activation cost for route $i$-$j$ (from route_fixed_costs.csv, table_id file_3_view_0, entry [i,j])
- $M_{ij}$: a sufficiently large constant for each $(i,j)$, e.g., $M_{ij} = \min(a_i, \sum_{j'} b_{j'})$

**Decision Variables:**
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$
- $y_{ij} \in \{0,1\}$: 1 if route $i$-$j$ is used, 0 otherwise

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} \left( c_{ij} x_{ij} + f_{ij} y_{ij} \right)
\]

**Subject to:**

1. **Supply capacity at each supplier:**
   \[
   \sum_{j \in J} x_{ij} \leq a_i \qquad \forall i \in I
   \]

2. **Demand satisfaction at each customer:**
   \[
   \sum_{i \in I} x_{ij} = b_j \qquad \forall j \in J
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

### Data Mapping

- $I$ (suppliers): file_0_view_0, column "Supplier"
- $J$ (customers): file_1_view_0, column "Customer"
- $a_i$: file_0_view_0, column "SupplyCapacity"
- $b_j$: file_1_view_0, column "Demand"
- $c_{ij}$: file_2_view_0, row "Supplier" $i$, column $j$
- $f_{ij}$: file_3_view_0, row "Supplier" $i$, column $j$
- $M_{ij}$: may use $a_i$ or $\sum_{j'} b_{j'}$ as a sufficiently large constant for each $(i,j)$

- $x_{ij}$: continuous, quantity shipped from $i$ to $j$
- $y_{ij}$: binary, 1 if route $i$-$j$ is used, 0 otherwise

---

**All sets, parameters, and variables are defined directly from the supplied CSV files as described above.**