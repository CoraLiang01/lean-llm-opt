#### Index Sets

- $G$: Set of raw grades (from 30-1.csv, column "Grade")
- $B$: Set of wine brands (from 30-2.csv, column "Brand")

#### Parameters

- $S_g$: Daily supply limit for raw grade $g \in G$ (from 30-1.csv, column "Daily Supply (kg)")
- $C_g$: Unit cost of raw grade $g \in G$ (from 30-1.csv, column "Cost (CNY/kg)")
- $P_b$: Selling price per kg of brand $b \in B$ (from 30-2.csv, column "Selling Price (CNY/kg)")
- $L_{b,g}$: Lower bound on the proportion of raw grade $g$ in brand $b$ (from 30-2.csv, column "Blending Requirements")
- $U_{b,g}$: Upper bound on the proportion of raw grade $g$ in brand $b$ (from 30-2.csv, column "Blending Requirements")

#### Decision Variables

- $x_{b,g} \geq 0$: Amount (kg) of raw grade $g$ used in brand $b$, for all $b \in B$, $g \in G$

#### Derived Quantities

- $y_b = \sum_{g \in G} x_{b,g}$: Total production (kg) of brand $b$

#### Objective

Maximize total net profit:
$$
\max \left[ \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \left( \sum_{b \in B} x_{b,g} \right) \right]
$$

#### Constraints

1. **Blending Requirements** (for all $b \in B$, $g \in G$ with specified bounds):

   $$
   L_{b,g} \cdot y_b \leq x_{b,g} \leq U_{b,g} \cdot y_b
   $$

   (If a lower or upper bound is not specified for a $(b,g)$ pair, omit the corresponding constraint.)

2. **Raw Material Supply** (for all $g \in G$):

   $$
   \sum_{b \in B} x_{b,g} \leq S_g
   $$

3. **Minimum Production for Red Brand**:

   $$
   y_{\text{Red}} \geq 2000
   $$

4. **Non-negativity**:

   $$
   x_{b,g} \geq 0 \quad \forall b \in B,\, g \in G
   $$

#### Data Mapping

- 30-1.csv (table_id: file_0_view_0): "Grade" $\rightarrow G$, "Daily Supply (kg)" $\rightarrow S_g$, "Cost (CNY/kg)" $\rightarrow C_g$
- 30-2.csv (table_id: file_1_view_0): "Brand" $\rightarrow B$, "Blending Requirements" $\rightarrow L_{b,g}, U_{b,g}$, "Selling Price (CNY/kg)" $\rightarrow P_b$