**Abstract Mathematical Model**

**Index Sets**
- $P$: Set of products (from columns A1, A2, ..., A80 in all tables)
- $T$: Set of time periods (days), $T = \{1, 2, ..., 22\}$

**Parameters**  
(For each product $p \in P$; all parameters indexed by product column name, e.g., $p = \text{A1}$)

- $d_p$: Maximum demand for product $p$ in 100 kg units (from 36-1.csv, row "Maximum Demand (100 kg units)")
- $q_p$: Maximum daily production quota for product $p$ in 100 kg units (from 36-1.csv, row "Production Quota (max per day)")
- $s_p$: Selling price per 100 kg for product $p$ (from 36-1.csv, row "Selling Price ($/100 kg)")
- $c_p$: Production cost per 100 kg for product $p$ (from 36-1.csv, row "Production Cost ($/100 kg)")
- $f_p$: Fixed activation cost for product $p$ (from 36-2.csv, row "Activation Cost ($)")
- $m_p$: Minimum batch size for product $p$ in 100 kg units (from 36-3.csv, row "Minimum Batch Size (100 kg units)")
- $D$: Number of production days in the month ($D = 22$)

**Decision Variables**
- $x_p \in \mathbb{Z}_{\geq 0}$: Total quantity of product $p$ to produce in the month (in 100 kg units)
- $y_p \in \{0,1\}$: 1 if production line for product $p$ is activated, 0 otherwise

**Objective Function**
\[
\max \sum_{p \in P} \left[ s_p \cdot x_p - c_p \cdot x_p - f_p \cdot y_p \right]
\]

**Constraints**

1. **Demand Constraint (do not exceed demand):**
   \[
   x_p \leq d_p \qquad \forall p \in P
   \]

2. **Monthly Production Quota Constraint (cannot exceed total monthly capacity):**
   \[
   x_p \leq q_p \cdot D \qquad \forall p \in P
   \]

3. **Minimum Batch Size Constraint (if produced, must meet minimum batch):**
   \[
   x_p \geq m_p \cdot y_p \qquad \forall p \in P
   \]

4. **Activation Constraint (cannot produce unless activated):**
   \[
   x_p \leq d_p \cdot y_p \qquad \forall p \in P
   \]
   (This ensures $x_p = 0$ if $y_p = 0$.)

5. **Variable Domains:**
   \[
   x_p \in \mathbb{Z}_{\geq 0}, \quad y_p \in \{0,1\} \qquad \forall p \in P
   \]

---

**Data Mapping**

- $P$: All product columns $\{\text{A1}, \text{A2}, ..., \text{A80}\}$ in all tables.
- $d_p$: 36-1.csv, table_id: file_0_view_0, row "Maximum Demand (100 kg units)", column $p$
- $q_p$: 36-1.csv, table_id: file_0_view_0, row "Production Quota (max per day)", column $p$
- $s_p$: 36-1.csv, table_id: file_0_view_0, row "Selling Price ($/100 kg)", column $p$
- $c_p$: 36-1.csv, table_id: file_0_view_0, row "Production Cost ($/100 kg)", column $p$
- $f_p$: 36-2.csv, table_id: file_1_view_0, row "Activation Cost ($)", column $p$
- $m_p$: 36-3.csv, table_id: file_2_view_0, row "Minimum Batch Size (100 kg units)", column $p$
- $D$: User query ($D = 22$)

**Notes**
- All variables and parameters are indexed by the product column names as given in the source files.
- All constraints and the objective are applied for every product $p$ in $P$.
- No additional constraints are imposed beyond those specified in the query and data.