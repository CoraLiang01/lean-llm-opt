#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified under ‘Aalop’ (from column ‘Product Name’).
- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in \mathcal{I}$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in \mathcal{I}$ (from column ‘Initial Inventory’).
- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill (decision variable).

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \geq 0, \quad \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Source file: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv
- Index set $\mathcal{I}$: All rows where ‘Product Name’ is classified under ‘Aalop’ (as per the user query; in this instance, all listed products).
- Parameter $A_i$: Column ‘Revenue’
- Parameter $d_i$: Column ‘Demand’
- Parameter $I_i$: Column ‘Initial Inventory’
- Decision variable $x_i$: Number of units of product $i$ to fulfill

All columns and rows are used as returned by the CSVQA tool, with no selection or abbreviation.