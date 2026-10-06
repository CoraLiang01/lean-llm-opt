**Mathematical Model (Capacitated Facility Location Problem):**

**Sets:**
- Let \( I \) be the set of plants, indexed by \( i \).  
  Data Mapping: \( I = \{\text{plant} \} \) from `file_0_view_0` column `"plant"`.
- Let \( J \) be the set of customers, indexed by \( j \).  
  Data Mapping: \( J = \{\text{customer} \} \) from `file_1_view_0` column `"customer"`.

**Parameters:**
- \( f_i \): Fixed opening cost for plant \( i \).  
  Data Mapping: `file_0_view_0`, column `"fixed_cost"`, row `"plant" = i"`.
- \( K_i \): Capacity of plant \( i \).  
  Data Mapping: `file_0_view_0`, column `"capacity"`, row `"plant" = i"`.
- \( d_j \): Demand of customer \( j \).  
  Data Mapping: `file_1_view_0`, column `"demand"`, row `"customer" = j"`.
- \( c_{ij} \): Per-unit transport cost from plant \( i \) to customer \( j \).  
  Data Mapping: `file_0_view_0`, column with name \( j \) (e.g., `"C1"`, `"C2"`, ...), row `"plant" = i"`.

**Decision Variables:**
- \( y_i \in \{0,1\} \): 1 if plant \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount shipped from plant \( i \) to customer \( j \).

**Objective:**
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand satisfaction:**  
   For all \( j \in J \):
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Plant capacity:**  
   For all \( i \in I \):
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i
   \]
3. **Variable domains:**  
   For all \( i \in I, j \in J \):
   \[
   x_{ij} \geq 0
   \]
   For all \( i \in I \):
   \[
   y_i \in \{0,1\}
   \]

---

**Data Mapping Summary:**

- \( I \): All `"plant"` values in `file_0_view_0` (cost.csv)
- \( J \): All `"customer"` values in `file_1_view_0` (demand.csv)
- \( f_i \): `file_0_view_0`, `"fixed_cost"`, row `"plant" = i"`
- \( K_i \): `file_0_view_0`, `"capacity"`, row `"plant" = i"`
- \( d_j \): `file_1_view_0`, `"demand"`, row `"customer" = j"`
- \( c_{ij} \): `file_0_view_0`, column \( j \) (e.g., `"C1"`, ...), row `"plant" = i"`

---

**Index Set Definitions:**
- \( I = \{\text{F1}, \text{F2}, \ldots, \text{F15}\} \) (from `file_0_view_0`)
- \( J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\} \) (from `file_1_view_0`)

---

**Summary:**  
This model determines which plants to open and how much to ship from each plant to each customer to minimize total cost, using all plant and customer data from the provided CSVs. All parameters are mapped directly to the source data as specified above.