#### Symbolic Mathematical Model

Let:
- $I$ = set of all dairy products (indexed by $i$)
- $A_i$ = revenue per unit of product $i$
- $d_i$ = deterministic demand for product $i$
- $s_i$ = initial inventory for product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Nonnegativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Integer domain, if required)}
\end{align*}
\]

#### Data Mapping

- Index set $I$: All unique values in column Full_Product_Name from table_id file_0_view_0.
- Parameter $A_i$: Revenue from column Revenue in table_id file_0_view_0, mapped by Full_Product_Name.
- Parameter $d_i$: Demand from column Demand in table_id file_0_view_0, mapped by Full_Product_Name.
- Parameter $s_i$: Initial Inventory from column Initial Inventory in table_id file_0_view_0, mapped by Full_Product_Name.
- Decision variable $x_i$: Number of units fulfilled for each $i \in I$.

All data is sourced from DairyGoodsSalesDataset.csv, table_id file_0_view_0, columns: Full_Product_Name, Revenue, Demand, Initial Inventory.