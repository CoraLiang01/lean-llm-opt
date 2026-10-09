#### Mathematical Optimization Model

**Index Sets:**
- $G$: Set of raw wine grades (from 30-1.csv, column "Grade")
- $B$: Set of wine brands (from 30-2.csv, column "Brand")

**Parameters:**
- $S_g$: Daily supply limit of grade $g \in G$ (from 30-1.csv, "Daily Supply (kg)")
- $C_g$: Unit cost of grade $g \in G$ (from 30-1.csv, "Cost (CNY/kg)")
- $P_b$: Selling price per kg of brand $b \in B$ (from 30-2.csv, "Selling Price (CNY/kg)")
- $L_{g,b}$: Lower bound on the proportion of grade $g$ in brand $b$ (from 30-2.csv, "Blending Requirements")
- $U_{g,b}$: Upper bound on the proportion of grade $g$ in brand $b$ (from 30-2.csv, "Blending Requirements")

**Decision Variables:**
- $x_{g,b} \geq 0$: Amount (kg) of grade $g$ used in brand $b$ (continuous, nonnegative)

**Auxiliary Expressions:**
- $y_b = \sum_{g \in G} x_{g,b}$: Total production (kg) of brand $b$

---

**Objective:**
\[
\max \left( \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \sum_{b \in B} x_{g,b} \right)
\]

---

**Constraints:**

1. **Blending Requirements:**  
   For each brand $b \in B$ and each grade $g \in G$ with a specified lower or upper bound:
   \[
   L_{g,b} \cdot y_b \leq x_{g,b} \leq U_{g,b} \cdot y_b
   \]
   (Apply only where a bound is specified; otherwise, omit.)

2. **Raw Material Supply:**
   \[
   \sum_{b \in B} x_{g,b} \leq S_g \qquad \forall g \in G
   \]

3. **Minimum Production for Red Brand:**
   \[
   y_{\text{Red}} \geq 2000
   \]

4. **Nonnegativity:**
   \[
   x_{g,b} \geq 0 \qquad \forall g \in G,\, b \in B
   \]

---

#### Data Mapping

- $G$ (grades): file_0_view_0, column "Grade"
- $B$ (brands): file_1_view_0, column "Brand"
- $S_g$: file_0_view_0, columns "Grade", "Daily Supply (kg)"
- $C_g$: file_0_view_0, columns "Grade", "Cost (CNY/kg)"
- $P_b$: file_1_view_0, columns "Brand", "Selling Price (CNY/kg)"
- $L_{g,b}$, $U_{g,b}$: file_1_view_0, columns "Brand", "Blending Requirements" (parse for each $g,b$ pair)
- $y_b$: defined as $\sum_{g \in G} x_{g,b}$

**Note:**  
- For each brand $b$, parse "Blending Requirements" to extract $L_{g,b}$ and $U_{g,b}$ for each grade $g$.
- If a lower or upper bound is not specified for a $(g,b)$ pair, do not impose a bound for that pair.

---

**Variable Domains:**  
- $x_{g,b} \geq 0$, continuous

---

**All index sets, parameters, and constraints are mapped directly from the provided CSV files as described above.**