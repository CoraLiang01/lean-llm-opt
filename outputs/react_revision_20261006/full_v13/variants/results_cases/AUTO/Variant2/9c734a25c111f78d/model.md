## Mixed-Integer Fixed-Charge Transportation Model

**Sets:**
- $I$: set of suppliers (from supplier_capacity.csv), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$
- $J$: set of customers (from customer_demand.csv), $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$

**Parameters:**
- $a_i$: supply capacity of supplier $i \in I$ (from supplier_capacity.csv, column SupplyCapacity, table_id file_0_view_0)
- $b_j$: demand of customer $j \in J$ (from customer_demand.csv, column Demand, table_id file_1_view_0)
- $c_{ij}$: per-unit transportation cost from supplier $i$ to customer $j$ (from route_variable_costs.csv, table_id file_2_view_0, columns C1–C12)
- $f_{ij}$: fixed route activation cost for shipping from $i$ to $j$ (from route_fixed_costs.csv, table_id file_3_view_0, columns C1–C12)
- $M_{ij}$: a sufficiently large constant for each $(i,j)$, e.g., $M_{ij} = a_i$ (or larger if needed)

**Decision Variables:**
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous)
- $y_{ij} \in \{0,1\}$: 1 if any positive amount is shipped from $i$ to $j$, 0 otherwise (binary)

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} \left( c_{ij} x_{ij} + f_{ij} y_{ij} \right)
\]

**Constraints:**

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
   (where $M_{ij}$ can be set to $a_i$ or a larger value if needed)

4. **Nonnegativity and integrality:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- **Suppliers $I$:** All values in column "Supplier" of supplier_capacity.csv (table_id file_0_view_0)
- **Customers $J$:** All values in column "Customer" of customer_demand.csv (table_id file_1_view_0)
- **$a_i$:** "SupplyCapacity" for each supplier $i$ in supplier_capacity.csv (file_0_view_0)
- **$b_j$:** "Demand" for each customer $j$ in customer_demand.csv (file_1_view_0)
- **$c_{ij}$:** Entry in route_variable_costs.csv (file_2_view_0) at row "Supplier" $i$, column $j$
- **$f_{ij}$:** Entry in route_fixed_costs.csv (file_3_view_0) at row "Supplier" $i$, column $j$
- **$M_{ij}$:** May be set as $a_i$ for each $(i,j)$

- **$x_{ij}$:** Decision variable, quantity shipped from $i$ to $j$
- **$y_{ij}$:** Decision variable, binary, 1 if route $i$-$j$ is used, 0 otherwise

---

This model ensures all customer demands are met, supplier capacities are not exceeded, and fixed route costs are incurred only if a route is used. All data is mapped directly from the provided CSV files as specified.