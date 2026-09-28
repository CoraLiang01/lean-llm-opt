#### Index Sets

- $W$: Set of widgets (from "product_resources.csv", column "Product"), $W = \{\text{Widget1}, \ldots, \text{Widget141}\}$

#### Parameters

- $a_w$: Labor hours required per unit of widget $w$ ("product_resources.csv", column "LaborHours")
- $b_w$: Material A required per unit of widget $w$ ("product_resources.csv", column "MaterialA")
- $c_w$: Material B required per unit of widget $w$ ("product_resources.csv", column "MaterialB")
- $p_w$: Base profit per unit of widget $w$ ("product_resources.csv", column "Profit")
- $L^{\max}$: Maximum available labor hours per month ("resource_limits.csv", row with Resource = "LaborHours", column "MonthlyLimit")
- $A^{\max}$: Maximum available Material A per month ("resource_limits.csv", row with Resource = "MaterialA", column "MonthlyLimit")
- $B^{\max}$: Maximum available Material B per month ("resource_limits.csv", row with Resource = "MaterialB", column "MonthlyLimit")
- $r$: CatalystX byproduct rate for Widget3 (5 kg per unit produced, scenario parameter)
- $P^{\text{cat}}$: Sale price per kg of CatalystX ($300$, scenario parameter)
- $D^{\text{cat}}$: Maximum CatalystX sales per month (1500 kg, scenario parameter)
- $C^{\text{cat}}$: Disposal cost per kg of unsold CatalystX ($200$, scenario parameter)

#### Decision Variables

- $x_w \geq 0$: Number of units of widget $w$ to produce (continuous or integer, as appropriate)
- $y \geq 0$: Amount (kg) of CatalystX sold

#### Objective Function

\[
\max \left\{ \sum_{w \in W} p_w x_w + P^{\text{cat}} y - C^{\text{cat}} \left( r x_{\text{Widget3}} - y \right) \right\}
\]

where $r x_{\text{Widget3}}$ is the total CatalystX generated, $y$ is the amount sold, and $r x_{\text{Widget3}} - y$ is the amount disposed.

#### Constraints

1. **Labor Hours Limit**
   \[
   \sum_{w \in W} a_w x_w \leq L^{\max}
   \]

2. **Material A Limit**
   \[
   \sum_{w \in W} b_w x_w \leq A^{\max}
   \]

3. **Material B Limit**
   \[
   \sum_{w \in W} c_w x_w \leq B^{\max}
   \]

4. **CatalystX Sales Cap**
   \[
   0 \leq y \leq \min\{ r x_{\text{Widget3}},\ D^{\text{cat}} \}
   \]

5. **Nonnegativity**
   \[
   x_w \geq 0 \quad \forall w \in W
   \]
   \[
   y \geq 0
   \]

#### Data Mapping

- Table: "product_resources.csv"
  - "Product" $\rightarrow$ $W$
  - "LaborHours" $\rightarrow$ $a_w$
  - "MaterialA" $\rightarrow$ $b_w$
  - "MaterialB" $\rightarrow$ $c_w$
  - "Profit" $\rightarrow$ $p_w$
- Table: "resource_limits.csv"
  - "Resource" = "LaborHours", "MonthlyLimit" $\rightarrow$ $L^{\max}$
  - "Resource" = "MaterialA", "MonthlyLimit" $\rightarrow$ $A^{\max}$
  - "Resource" = "MaterialB", "MonthlyLimit" $\rightarrow$ $B^{\max}$

- Scenario parameters (from user description, not CSV): $r$, $P^{\text{cat}}$, $D^{\text{cat}}$, $C^{\text{cat}}$ as defined above.

---

This is a complete abstract mathematical model as requested.