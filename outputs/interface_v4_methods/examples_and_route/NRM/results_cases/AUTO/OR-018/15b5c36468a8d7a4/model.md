ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of Baby products, indexed by $i$ (from all rows in Salesdata.csv where Product Name starts with "Baby")

Parameters:
- $r_i$: Revenue per unit of product $i$ (Salesdata.csv, column Revenue, table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (Salesdata.csv, column Demand, table_id: file_0_view_0)
- $s_i$: Initial Inventory for product $i$ (Salesdata.csv, column Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$

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
- $I$: All rows in Salesdata.csv where Product Name starts with "Baby" (table_id: file_0_view_0)
- $r_i$: Salesdata.csv, column Revenue, table_id: file_0_view_0
- $d_i$: Salesdata.csv, column Demand, table_id: file_0_view_0
- $s_i$: Salesdata.csv, column Initial Inventory, table_id: file_0_view_0