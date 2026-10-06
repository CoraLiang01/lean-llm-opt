ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of products classified as ‘27in’, indexed by $i$ (see Data Mapping).

Parameters:
- $r_i$: Revenue per unit of product $i$ (from Revenue column).
- $d_i$: Demand for product $i$ (from Demand column).
- $s_i$: Initial Inventory of product $i$ (from Initial Inventory column).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
x_i \leq d_i \qquad \forall i \in I \tag{Demand constraint}
\]
\[
x_i \leq s_i \qquad \forall i \in I \tag{Inventory constraint}
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

Data Mapping:

- $I$: All rows in SalesDataAnalysis.csv where Product Name starts with "27in".
- $r_i$: SalesDataAnalysis.csv, column "Revenue", for each $i$.
- $d_i$: SalesDataAnalysis.csv, column "Demand", for each $i$.
- $s_i$: SalesDataAnalysis.csv, column "Initial Inventory", for each $i$.

Business Identifier: Product Name (SalesDataAnalysis.csv, column "Product Name").