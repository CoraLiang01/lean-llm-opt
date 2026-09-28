#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of suppliers (indexed by $i$)
- $J$: set of supermarkets (indexed by $j$)

**Parameters:**
- $f_i$: fixed cost to activate supplier $i$  
  (Data: table_id = file_1_view_0, column = fixed_costs)
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$  
  (Data: table_id = file_2_view_0, columns = [C1, C2], row_id = Unnamed: 0)
- $d_j$: demand of supermarket $j$  
  (Data: table_id = file_0_view_0, column = demand)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$

**Objective:**
\[
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand Satisfaction:**  
  $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Supplier Activation:**  
  $\sum_{j \in J} x_{ij} \leq \left(\sum_{j \in J} d_j\right) y_i \quad \forall i \in I$

3. **Variable Domains:**  
  $y_i \in \{0,1\} \quad \forall i \in I$  
  $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$

---

**Data Mapping:**

- $I$ (suppliers): file_1_view_0, column = Unnamed: 0
- $J$ (supermarkets): file_0_view_0, column = customer
- $f_i$: file_1_view_0, column = fixed_costs
- $c_{ij}$: file_2_view_0, columns = [C1, C2], row_id = Unnamed: 0
- $d_j$: file_0_view_0, column = demand