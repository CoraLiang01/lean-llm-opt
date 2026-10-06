**Mathematical Optimization Model**

---

**Index Sets:**
- $G$: Set of raw wine grades (from 30-1.csv, column "Grade")
- $B$: Set of wine brands (from 30-2.csv, column "Brand")

**Parameters:**
- $S_g$: Daily supply limit of grade $g$ (from 30-1.csv, "Daily Supply (kg)")
- $C_g$: Unit cost of grade $g$ (from 30-1.csv, "Cost (CNY/kg)")
- $P_b$: Selling price per kg of brand $b$ (from 30-2.csv, "Selling Price (CNY/kg)")
- $L_{g,b}$: Lower bound on proportion of grade $g$ in brand $b$ (from 30-2.csv, "Blending Requirements")
- $U_{g,b}$: Upper bound on proportion of grade $g$ in brand $b$ (from 30-2.csv, "Blending Requirements")

**Decision Variables:**
- $x_{g,b} \geq 0$: Amount (kg) of grade $g$ used in brand $b$ production

**Auxiliary Variables:**
- $y_b = \sum_{g \in G} x_{g,b}$: Total production (kg) of brand $b$

---

**Objective:**
\[
\max \left[ \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \left( \sum_{b \in B} x_{g,b} \right) \right]
\]

---

**Constraints:**

1. **Raw Material Supply Limits:**
   \[
   \sum_{b \in B} x_{g,b} \leq S_g \qquad \forall g \in G
   \]

2. **Blending Requirements:**
   For each $(g, b)$ where a lower bound $L_{g,b}$ is specified:
   \[
   x_{g,b} \geq L_{g,b} \cdot y_b
   \]
   For each $(g, b)$ where an upper bound $U_{g,b}$ is specified:
   \[
   x_{g,b} \leq U_{g,b} \cdot y_b
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

**Data Mapping**

- $G$ (grades): file_0_view_0, column "Grade"
- $S_g$: file_0_view_0, column "Daily Supply (kg)", keyed by "Grade"
- $C_g$: file_0_view_0, column "Cost (CNY/kg)", keyed by "Grade"
- $B$ (brands): file_1_view_0, column "Brand"
- $P_b$: file_1_view_0, column "Selling Price (CNY/kg)", keyed by "Brand"
- $L_{g,b}$, $U_{g,b}$: file_1_view_0, column "Blending Requirements", parsed for each $(g, b)$ pair

---

**Blending Requirements Mapping (from file_1_view_0, "Blending Requirements"):**

- For brand "Red":
    - $U_{\text{I},\text{Red}} = 0.10$
    - $L_{\text{II},\text{Red}} = 0.50$
- For brand "Yellow":
    - $U_{\text{III},\text{Yellow}} = 0.70$
    - $L_{\text{I},\text{Yellow}} = 0.20$
- For brand "Blue":
    - $U_{\text{I},\text{Blue}} = 0.50$
    - $L_{\text{II},\text{Blue}} = 0.10$

---

**Summary:**  
Maximize total net profit by choosing $x_{g,b}$ (kg of grade $g$ in brand $b$), subject to raw material supply, blending proportion bounds, and minimum Red brand production, using all parameters and constraints as mapped above.