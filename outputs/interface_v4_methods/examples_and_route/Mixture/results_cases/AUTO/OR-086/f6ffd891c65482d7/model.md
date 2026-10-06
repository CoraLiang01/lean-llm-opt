#### Abstract Mathematical Model

Let:
- $G$ = set of raw grades (from 30-1.csv, column "Grade")
- $B$ = set of wine brands (from 30-2.csv, column "Brand")

Parameters:
- $S_g$ = daily supply limit of grade $g$ (from 30-1.csv, "Daily Supply (kg)")
- $C_g$ = unit cost of grade $g$ (from 30-1.csv, "Cost (CNY/kg)")
- $P_b$ = selling price per kg of brand $b$ (from 30-2.csv, "Selling Price (CNY/kg)")
- $L_{g,b}$ = lower bound on proportion of grade $g$ in brand $b$ (from 30-2.csv, "Blending Requirements")
- $U_{g,b}$ = upper bound on proportion of grade $g$ in brand $b$ (from 30-2.csv, "Blending Requirements")

Decision Variables:
- $x_{g,b} \geq 0$: amount (kg) of grade $g$ used in brand $b$ (continuous, nonnegative)

Auxiliary:
- $y_b = \sum_{g \in G} x_{g,b}$: total production (kg) of brand $b$

Objective:
\[
\max \left( \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \sum_{b \in B} x_{g,b} \right)
\]

Subject to:

1. **Blending Requirements** (for all $b \in B$, $g \in G$ with specified bounds):

   For each $(g,b)$ with a lower bound $L_{g,b}$:
   \[
   x_{g,b} \geq L_{g,b} \cdot y_b
   \]
   For each $(g,b)$ with an upper bound $U_{g,b}$:
   \[
   x_{g,b} \leq U_{g,b} \cdot y_b
   \]

2. **Raw Material Supply** (for all $g \in G$):
\[
\sum_{b \in B} x_{g,b} \leq S_g
\]

3. **Minimum Production for Red Brand**:
\[
y_{\text{Red}} \geq 2000
\]

4. **Nonnegativity**:
\[
x_{g,b} \geq 0 \quad \forall g \in G, b \in B
\]

---

#### Data Mapping

- $G$ (grades): file_0_view_0, column "Grade"
- $S_g$: file_0_view_0, columns "Grade", "Daily Supply (kg)"
- $C_g$: file_0_view_0, columns "Grade", "Cost (CNY/kg)"
- $B$ (brands): file_1_view_0, column "Brand"
- $P_b$: file_1_view_0, columns "Brand", "Selling Price (CNY/kg)"
- $L_{g,b}$, $U_{g,b}$: file_1_view_0, columns "Brand", "Blending Requirements" (parse for each brand-grade pair)

---

#### Blending Requirements Mapping (from file_1_view_0, "Blending Requirements"):

- For Red:
  - Grade I: $U_{\text{I,Red}} = 0.10$
  - Grade II: $L_{\text{II,Red}} = 0.50$
- For Yellow:
  - Grade III: $U_{\text{III,Yellow}} = 0.70$
  - Grade I: $L_{\text{I,Yellow}} = 0.20$
- For Blue:
  - Grade I: $U_{\text{I,Blue}} = 0.50$
  - Grade II: $L_{\text{II,Blue}} = 0.10$

All other $L_{g,b}$ and $U_{g,b}$ are undefined (no constraint).

---

All parameters and sets are mapped directly from the supplied CSV files and columns as above.