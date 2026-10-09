#### Index Sets

- $G$: Set of raw wine grades (from 30-1.csv, column "Grade")
- $B$: Set of wine brands (from 30-2.csv, column "Brand")

#### Parameters

From 30-1.csv (table_id: file_0_view_0, columns: "Grade", "Daily Supply (kg)", "Cost (CNY/kg)"):

- $S_g$: Daily supply limit of grade $g \in G$ ("Daily Supply (kg)")
- $C_g$: Unit cost of grade $g \in G$ ("Cost (CNY/kg)")

From 30-2.csv (table_id: file_1_view_0, columns: "Brand", "Blending Requirements", "Selling Price (CNY/kg)"):

- $P_b$: Selling price per kg of brand $b \in B$ ("Selling Price (CNY/kg)")
- $L_{g,b}$: Lower bound on the proportion of grade $g$ in brand $b$ (from "Blending Requirements")
- $U_{g,b}$: Upper bound on the proportion of grade $g$ in brand $b$ (from "Blending Requirements")

#### Decision Variables

- $x_{g,b} \geq 0$: Amount (kg) of grade $g \in G$ used in brand $b \in B$

#### Derived Quantities

- $y_b = \sum_{g \in G} x_{g,b}$: Total production (kg) of brand $b \in B$

#### Objective

Maximize total net profit:
$$
\max \left( \sum_{b \in B} P_b \cdot y_b - \sum_{g \in G} C_g \cdot \sum_{b \in B} x_{g,b} \right)
$$

#### Constraints

1. **Blending Requirements** (for all $b \in B$, $g \in G$ with specified bounds):

   $$
   L_{g,b} \cdot y_b \leq x_{g,b} \leq U_{g,b} \cdot y_b
   $$

   (If a lower or upper bound is not specified for $(g,b)$, omit that bound.)

2. **Raw Material Supply** (for all $g \in G$):

   $$
   \sum_{b \in B} x_{g,b} \leq S_g
   $$

3. **Minimum Production for Red Brand**:

   $$
   y_{\text{Red}} \geq 2000
   $$

4. **Nonnegativity**:

   $$
   x_{g,b} \geq 0 \quad \forall g \in G,\, b \in B
   $$

#### Data Mapping

- $G$, $S_g$, $C_g$: from 30-1.csv (table_id: file_0_view_0), columns "Grade", "Daily Supply (kg)", "Cost (CNY/kg)"
- $B$, $P_b$, $L_{g,b}$, $U_{g,b}$: from 30-2.csv (table_id: file_1_view_0), columns "Brand", "Selling Price (CNY/kg)", "Blending Requirements"
- Blending requirements for each $(g,b)$ pair are mapped directly from the "Blending Requirements" column in 30-2.csv (table_id: file_1_view_0), using the exact text and bounds as specified in each record.