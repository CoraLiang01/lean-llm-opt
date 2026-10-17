#### Index Sets

- $G$: set of raw wine grades (from 30-1.csv, column "Grade")
- $B$: set of wine brands (from 30-2.csv, column "Brand")

#### Parameters

- $S_g$: daily supply limit of raw grade $g \in G$ (from 30-1.csv, column "Daily Supply (kg)")
- $C_g$: unit cost of raw grade $g \in G$ (from 30-1.csv, column "Cost (CNY/kg)")
- $P_b$: selling price per kg of brand $b \in B$ (from 30-2.csv, column "Selling Price (CNY/kg)")
- $L_{b,g}$: lower bound on the proportion of grade $g$ in brand $b$ (from 30-2.csv, column "Blending Requirements", parsed per brand and grade; if not specified, set to 0)
- $U_{b,g}$: upper bound on the proportion of grade $g$ in brand $b$ (from 30-2.csv, column "Blending Requirements", parsed per brand and grade; if not specified, set to 1)

#### Decision Variables

- $x_{b,g} \geq 0$: amount (kg) of raw grade $g$ used in brand $b$, for all $b \in B$, $g \in G$

#### Derived Quantities

- $y_b = \sum_{g \in G} x_{b,g}$: total production (kg) of brand $b$

#### Objective

Maximize total net profit:
$$
\max \left[ \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \left( \sum_{b \in B} x_{b,g} \right) \right]
$$

#### Constraints

1. **Blending Requirements** (for all $b \in B$, $g \in G$):

   $$
   L_{b,g} \cdot y_b \leq x_{b,g} \leq U_{b,g} \cdot y_b
   $$

2. **Raw Material Supply** (for all $g \in G$):

   $$
   \sum_{b \in B} x_{b,g} \leq S_g
   $$

3. **Minimum Production for Red Brand**:

   $$
   y_{\text{Red}} \geq 2000
   $$

4. **Nonnegativity**:

   $$
   x_{b,g} \geq 0 \quad \forall b \in B,\, g \in G
   $$

#### Data Mapping

- Table `file_0_view_0` (30-1.csv):  
  - $G$ = all values in column "Grade"
  - $S_g$ = column "Daily Supply (kg)", per grade $g$
  - $C_g$ = column "Cost (CNY/kg)", per grade $g$
- Table `file_1_view_0` (30-2.csv):  
  - $B$ = all values in column "Brand"
  - $P_b$ = column "Selling Price (CNY/kg)", per brand $b$
  - $L_{b,g}$, $U_{b,g}$ = parsed from column "Blending Requirements", per brand $b$ and grade $g$
- Table `file_1_view_1` (30-2.csv, filtered where Brand = "Red"):  
  - Minimum production constraint for $y_{\text{Red}}$

All data is used as returned by CSVQA, with blending requirements parsed per brand and grade as specified in the source.