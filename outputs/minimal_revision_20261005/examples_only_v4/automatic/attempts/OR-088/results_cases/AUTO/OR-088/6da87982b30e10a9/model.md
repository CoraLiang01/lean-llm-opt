**Mathematical Model (Capacitated Facility Location Problem):**

**Sets:**
- Let \( I \) be the set of plants, indexed by \( i \), where \( I = \{\text{F1}, \ldots, \text{F15}\} \) (from cost.csv, column plant, table_id: file_0_view_0).
- Let \( J \) be the set of customers, indexed by \( j \), where \( J = \{\text{C1}, \ldots, \text{C15}\} \) (from demand.csv, column customer, table_id: file_1_view_0).

**Parameters:**
- \( f_i \): Fixed opening cost for plant \( i \) (from cost.csv, column fixed_cost, table_id: file_0_view_0, row plant = \( i \)).
- \( K_i \): Capacity of plant \( i \) (from cost.csv, column capacity, table_id: file_0_view_0, row plant = \( i \)).
- \( d_j \): Demand of customer \( j \) (from demand.csv, column demand, table_id: file_1_view_0, row customer = \( j \)).
- \( c_{ij} \): Per-unit transport cost from plant \( i \) to customer \( j \) (from cost.csv, column \( j \), table_id: file_0_view_0, row plant = \( i \)).

**Decision Variables:**
- \( y_i \in \{0,1\} \): 1 if plant \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount shipped from plant \( i \) to customer \( j \).

**Objective:**
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]
2. **Plant capacity (only if opened):**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I
   \]
3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
4. **Binary plant opening:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

**Data Mapping:**

- \( I \): All values in column plant of table_id: file_0_view_0 (cost.csv)
- \( J \): All values in column customer of table_id: file_1_view_0 (demand.csv)
- \( f_i \): Value in column fixed_cost of table_id: file_0_view_0, row where plant = \( i \)
- \( K_i \): Value in column capacity of table_id: file_0_view_0, row where plant = \( i \)
- \( d_j \): Value in column demand of table_id: file_1_view_0, row where customer = \( j \)
- \( c_{ij} \): Value in column \( j \) (e.g., C1, C2, ..., C15) of table_id: file_0_view_0, row where plant = \( i \)

---

**Variable Domains:**
- \( y_i \in \{0,1\} \) for all \( i \in I \)
- \( x_{ij} \geq 0 \) for all \( i \in I,\, j \in J \)