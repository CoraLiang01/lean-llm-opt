## Symbolic Mathematical Model

### Sets
- $P = \{\text{I}, \text{II}, \text{III}\}$: Products
- $A = \{\text{A1}, \text{A2}\}$: Equipment for procedure A
- $B = \{\text{B1}, \text{B2}, \text{B3}\}$: Equipment for procedure B

### Parameters (from Data Mapping)
Let $t_{e,p}$ = processing time (hours/unit) for product $p$ on equipment $e$ (from "Product I", "Product II", "Product III" columns, table_id: file_0_view_0)
Let $T_e$ = available operating time (hours) for equipment $e$ (from "Available Equipment Operating Time", table_id: file_0_view_0)
Let $C_e$ = equipment cost at full load (yuan) for equipment $e$ (from "Equipment Cost at Full Load (yuan)", table_id: file_0_view_0)
Let $c_p$ = raw material cost per unit for product $p$ (from "Raw Material Cost (yuan/unit)", table_id: file_0_view_0, row "Raw Material Cost (yuan/unit)")
Let $s_p$ = selling price per unit for product $p$ (from "Unit Price (yuan/unit)", table_id: file_0_view_0, row "Unit Price (yuan/unit)")

### Decision Variables
- $x_{e,p} \geq 0$: quantity of product $p$ processed on equipment $e$ (continuous)

### Feasibility Sets (from process restrictions)
- For procedure A:
    - $A_p$ = allowed A-equipment for product $p$:
        - $A_{\text{I}} = \{\text{A1}, \text{A2}\}$
        - $A_{\text{II}} = \{\text{A1}, \text{A2}\}$
        - $A_{\text{III}} = \{\text{A2}\}$
- For procedure B:
    - $B_p$ = allowed B-equipment for product $p$:
        - $B_{\text{I}} = \{\text{B1}, \text{B2}, \text{B3}\}$
        - $B_{\text{II}} = \{\text{B1}\}$
        - $B_{\text{III}} = \{\text{B2}\}$

### Objective Function
Maximize total profit:
\[
\max \left\{
\sum_{p \in P} \left[ s_p \cdot y_p - c_p \cdot y_p \right]
- \sum_{e \in A \cup B} \frac{C_e}{T_e} \cdot \left( \sum_{p \in P_e} t_{e,p} x_{e,p} \right)
\right\}
\]
where:
- $y_p$ = total production of product $p$ (units)
- $P_e$ = set of products that can be processed on equipment $e$ (from data, i.e., non-empty $t_{e,p}$)
- $t_{e,p}$, $C_e$, $T_e$ as above

### Constraints

1. **Production Consistency (each product's output must be the same after both procedures):**
   - For all $p \in P$:
     \[
     \sum_{e \in A_p} x_{e,p} = \sum_{e' \in B_p} x_{e',p} = y_p
     \]

2. **Equipment Time Limits:**
   - For all $e \in A \cup B$:
     \[
     \sum_{p \in P_e} t_{e,p} x_{e,p} \leq T_e
     \]
     where $P_e$ is the set of products $p$ for which $t_{e,p}$ is defined (i.e., not blank).

3. **Non-negativity:**
   - $x_{e,p} \geq 0$ for all $e,p$ in their allowed sets.

### Data Mapping

- $t_{e,p}$: file_0_view_0, columns "Product I", "Product II", "Product III", rows with equipment $e$
- $T_e$: file_0_view_0, column "Available Equipment Operating Time", row with equipment $e$
- $C_e$: file_0_view_0, column "Equipment Cost at Full Load (yuan)", row with equipment $e$
- $c_p$: file_0_view_0, row "Raw Material Cost (yuan/unit)", columns "Product I", "Product II", "Product III"
- $s_p$: file_0_view_0, row "Unit Price (yuan/unit)", columns "Product I", "Product II", "Product III"

### Explicit Model (with index sets and data mapping)

Let $A = \{\text{A1}, \text{A2}\}$, $B = \{\text{B1}, \text{B2}, \text{B3}\}$, $P = \{\text{I}, \text{II}, \text{III}\}$

Decision variables:
- $x_{e,p} \geq 0$ for $e \in A_p$ (procedure A), $e \in B_p$ (procedure B)
- $y_p \geq 0$ for $p \in P$

Objective:
\[
\max \left\{
\sum_{p \in P} (s_p - c_p) y_p
- \sum_{e \in A \cup B} \frac{C_e}{T_e} \left( \sum_{p \in P_e} t_{e,p} x_{e,p} \right)
\right\}
\]

Subject to:
\[
\forall p \in P: \quad \sum_{e \in A_p} x_{e,p} = \sum_{e' \in B_p} x_{e',p} = y_p
\]
\[
\forall e \in A \cup B: \quad \sum_{p \in P_e} t_{e,p} x_{e,p} \leq T_e
\]
\[
x_{e,p} \geq 0, \quad y_p \geq 0
\]

**Data Mapping:** All parameters are mapped to file_0_view_0 as described above. All index sets are defined by the non-empty entries in the table.

---

This model maximizes profit by optimally assigning production quantities to allowed equipment, respecting all process, time, and cost constraints, with all data and index sets mapped directly to the provided CSV.