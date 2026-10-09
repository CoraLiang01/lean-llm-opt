### Mathematical Model

**Sets:**
- $G = \{\text{I}, \text{II}, \text{III}\}$: Raw grades (from 30-1.csv, column "Grade")
- $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$: Wine brands (from 30-2.csv, column "Brand")

**Parameters:**
- $S_g$: Daily supply limit of grade $g$ (from 30-1.csv, "Daily Supply (kg)")
- $C_g$: Unit cost of grade $g$ (from 30-1.csv, "Cost (CNY/kg)")
- $P_b$: Selling price of brand $b$ (from 30-2.csv, "Selling Price (CNY/kg)")
- Blending requirements for each $(b,g)$ as per 30-2.csv "Blending Requirements" (see Data Mapping)

**Decision Variables:**
- $x_{b,g} \geq 0$: Amount (kg) of grade $g$ used in brand $b$

**Auxiliary:**
- $y_b = \sum_{g \in G} x_{b,g}$: Total production (kg) of brand $b$

**Objective:**
\[
\max \left[ \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \left( \sum_{b \in B} x_{b,g} \right) \right]
\]

**Constraints:**

1. **Raw Material Supply:**
   \[
   \sum_{b \in B} x_{b,g} \leq S_g \quad \forall g \in G
   \]

2. **Blending Requirements:**

   - For Red:
     - $\frac{x_{\text{Red},\text{I}}}{y_{\text{Red}}} < 0.10$  (I less than 10%)
     - $\frac{x_{\text{Red},\text{II}}}{y_{\text{Red}}} > 0.50$  (II more than 50%)
   - For Yellow:
     - $\frac{x_{\text{Yellow},\text{III}}}{y_{\text{Yellow}}} < 0.70$  (III less than 70%)
     - $\frac{x_{\text{Yellow},\text{I}}}{y_{\text{Yellow}}} > 0.20$  (I more than 20%)
   - For Blue:
     - $\frac{x_{\text{Blue},\text{I}}}{y_{\text{Blue}}} < 0.50$  (I less than 50%)
     - $\frac{x_{\text{Blue},\text{II}}}{y_{\text{Blue}}} > 0.10$  (II more than 10%)

   For any $y_b = 0$, set $x_{b,g} = 0$ for all $g$.

3. **Minimum Production for Red:**
   \[
   y_{\text{Red}} \geq 2000
   \]

4. **Non-negativity:**
   \[
   x_{b,g} \geq 0 \quad \forall b \in B,\, g \in G
   \]

---

### Data Mapping

- **file_0_view_0** (30-1.csv): $G$, $S_g$, $C_g$
  - "Grade" $\rightarrow$ $G$
  - "Daily Supply (kg)" $\rightarrow$ $S_g$
  - "Cost (CNY/kg)" $\rightarrow$ $C_g$
- **file_1_view_0** (30-2.csv): $B$, $P_b$, blending requirements
  - "Brand" $\rightarrow$ $B$
  - "Selling Price (CNY/kg)" $\rightarrow$ $P_b$
  - "Blending Requirements" $\rightarrow$ constraints above

---

**All sets, parameters, and constraints are defined directly from the current CSV data.**