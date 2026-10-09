### Mathematical Model

**Index Sets:**
- $I$: Set of widgets (from file_1_view_0, column Product)
- $R$: Set of resources $\{\text{LaborHours}, \text{MaterialA}, \text{MaterialB}\}$ (from file_0_view_0, column Resource)

**Parameters:**
- $a_{i,r}$: Amount of resource $r \in R$ consumed per unit of widget $i \in I$ (from file_1_view_0, columns LaborHours, MaterialA, MaterialB)
- $b_r$: Monthly limit of resource $r \in R$ (from file_0_view_0, column MonthlyLimit)
- $p_i$: Base profit per unit of widget $i \in I$ (from file_1_view_0, column Profit)
- $q$: CatalystX generated per unit of Widget3 (5 kg/unit, query-defined)
- $v_{\text{sell}}$: Sale price per kg of CatalystX ($300$, query-defined)
- $v_{\text{dispose}}$: Disposal cost per kg of CatalystX ($200$, query-defined)
- $S$: Maximum CatalystX sales per month (1500 kg, query-defined)

**Decision Variables:**
- $x_i \geq 0$: Number of units of widget $i \in I$ to produce (continuous, as not specified integer)
- $y \geq 0$: Amount (kg) of CatalystX to sell (continuous)

**Objective:**
\[
\max \left[ \sum_{i \in I} p_i x_i + v_{\text{sell}} y - v_{\text{dispose}} \left( q x_{\text{Widget3}} - y \right) \right]
\]
where $x_{\text{Widget3}}$ is the production quantity of Widget3.

**Constraints:**
1. Resource limits:
   \[
   \sum_{i \in I} a_{i,r} x_i \leq b_r \qquad \forall r \in R
   \]
2. CatalystX sales cannot exceed production:
   \[
   y \leq q x_{\text{Widget3}}
   \]
3. CatalystX sales capped by market:
   \[
   y \leq S
   \]
4. Nonnegativity:
   \[
   x_i \geq 0 \qquad \forall i \in I
   \]
   \[
   y \geq 0
   \]

---

### Data Mapping

- $I$: file_1_view_0, column Product
- $R$: file_0_view_0, column Resource
- $a_{i,\text{LaborHours}}$: file_1_view_0, column LaborHours, for each $i$
- $a_{i,\text{MaterialA}}$: file_1_view_0, column MaterialA, for each $i$
- $a_{i,\text{MaterialB}}$: file_1_view_0, column MaterialB, for each $i$
- $b_r$: file_0_view_0, column MonthlyLimit, for each $r$
- $p_i$: file_1_view_0, column Profit, for each $i$
- $q$: 5 (query-defined, for Widget3 only)
- $v_{\text{sell}}$: 300 (query-defined)
- $v_{\text{dispose}}$: 200 (query-defined)
- $S$: 1500 (query-defined)
- $x_i$: production quantity of widget $i$ (decision variable)
- $y$: amount of CatalystX sold (decision variable)
- $x_{\text{Widget3}}$: $x_i$ for $i$ such that Product = "Widget3" (file_1_view_0, Product)

---

**All parameters and index sets are mapped directly from the returned CSV data or the user query.**