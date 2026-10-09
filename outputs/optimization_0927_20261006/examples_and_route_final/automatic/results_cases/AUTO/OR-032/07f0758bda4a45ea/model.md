Let $x_i$ be the number of units of product $i$ (where $i$ indexes the five ‘Books’ products below) to fulfill.

##### Sets and Parameters

- Products $i$:
  - Books_15.15: Revenue = 15.15, Demand = 1980, Initial Inventory = 9920.0
  - Books_30.3: Revenue = 30.3, Demand = 3024, Initial Inventory = 20160.0
  - Books_45.45: Revenue = 45.45, Demand = 4536, Initial Inventory = 30000.0
  - Books_60.6: Revenue = 60.6, Demand = 5601, Initial Inventory = 38360.0
  - Books_75.75: Revenue = 75.75, Demand = 7567, Initial Inventory = 51450.0

##### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$ (number of units of product $i$ to fulfill)

##### Objective

\[
\max \quad 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}}
\]

##### Constraints

For each product $i$:

- Demand constraint:
  \[
  x_i \leq \text{Demand}_i
  \]
- Inventory constraint:
  \[
  x_i \leq \text{Initial Inventory}_i
  \]
- Nonnegativity and integrality:
  \[
  x_i \in \mathbb{Z}_{\geq 0}
  \]

##### Explicitly, for each product:

\[
\begin{align*}
x_{\text{Books\_15.15}} &\leq 1980 \\
x_{\text{Books\_15.15}} &\leq 9920.0 \\
x_{\text{Books\_30.3}} &\leq 3024 \\
x_{\text{Books\_30.3}} &\leq 20160.0 \\
x_{\text{Books\_45.45}} &\leq 4536 \\
x_{\text{Books\_45.45}} &\leq 30000.0 \\
x_{\text{Books\_60.6}} &\leq 5601 \\
x_{\text{Books\_60.6}} &\leq 38360.0 \\
x_{\text{Books\_75.75}} &\leq 7567 \\
x_{\text{Books\_75.75}} &\leq 51450.0 \\
x_i &\in \mathbb{Z}_{\geq 0}, \quad \forall i
\end{align*}
\]

##### Summary

Maximize total revenue from fulfilling ‘Books’ products, subject to demand and inventory limits for each product, with nonnegative integer fulfillment decisions.