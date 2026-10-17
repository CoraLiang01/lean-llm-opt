#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of widgets, $I = \{\text{Widget1}, \ldots, \text{Widget141}\}$

**Parameters:**
- $a_i$: Labor hours required per unit of widget $i$ (from product_resources.csv, column "LaborHours")
- $b_i$: Material A required per unit of widget $i$ (from product_resources.csv, column "MaterialA")
- $c_i$: Material B required per unit of widget $i$ (from product_resources.csv, column "MaterialB")
- $p_i$: Base profit per unit of widget $i$ (from product_resources.csv, column "Profit")
- $L$: Monthly labor hour limit (from resource_limits.csv, row where Resource = "LaborHours", column "MonthlyLimit")
- $M_A$: Monthly Material A limit (from resource_limits.csv, row where Resource = "MaterialA", column "MonthlyLimit")
- $M_B$: Monthly Material B limit (from resource_limits.csv, row where Resource = "MaterialB", column "MonthlyLimit")
- $r$: CatalystX byproduct rate for Widget3 ($r = 5$ kg per unit of Widget3 produced)
- $v$: Sale price per kg of CatalystX ($v = 300$)
- $d$: Disposal cost per kg of unsold CatalystX ($d = 200$)
- $S$: Maximum CatalystX sales per month ($S = 1500$ kg)

**Decision Variables:**
- $x_i \geq 0$: Number of units of widget $i$ to produce (continuous or integer, as appropriate)
- $y \geq 0$: Amount (kg) of CatalystX sold to pharmaceutical partners
- $z \geq 0$: Amount (kg) of CatalystX disposed as hazardous waste

**Objective Function:**
\[
\max \left\{ \sum_{i \in I} p_i x_i + v y - d z \right\}
\]

**Constraints:**

1. **Labor Hours:**
   \[
   \sum_{i \in I} a_i x_i \leq L
   \]

2. **Material A:**
   \[
   \sum_{i \in I} b_i x_i \leq M_A
   \]

3. **Material B:**
   \[
   \sum_{i \in I} c_i x_i \leq M_B
   \]

4. **CatalystX Byproduct Balance (Widget3 only):**
   \[
   y + z = r \cdot x_{\text{Widget3}}
   \]

5. **CatalystX Sales Cap:**
   \[
   y \leq S
   \]

6. **Nonnegativity:**
   \[
   x_i \geq 0 \quad \forall i \in I
   \]
   \[
   y \geq 0
   \]
   \[
   z \geq 0
   \]

---

#### Data Mapping

- **product_resources.csv** (table_id: file_1_view_0)
  - "Product" $\rightarrow$ $i \in I$
  - "LaborHours" $\rightarrow$ $a_i$
  - "MaterialA" $\rightarrow$ $b_i$
  - "MaterialB" $\rightarrow$ $c_i$
  - "Profit" $\rightarrow$ $p_i$
- **resource_limits.csv** (table_id: file_0_view_0)
  - "Resource" = "LaborHours", "MonthlyLimit" $\rightarrow$ $L$
  - "Resource" = "MaterialA", "MonthlyLimit" $\rightarrow$ $M_A$
  - "Resource" = "MaterialB", "MonthlyLimit" $\rightarrow$ $M_B$

- CatalystX byproduct and market/disposal parameters ($r$, $v$, $d$, $S$) are scenario constants from the user query.

---

**Note:** All widgets are eligible for production. Only Widget3 generates CatalystX as a byproduct. The model maximizes total profit, accounting for both the base profit from widgets and the net value (sale or disposal) of CatalystX.