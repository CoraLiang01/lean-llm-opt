**Mathematical Model**

Let:
- $I$ = index set of all car models with identifier 'FDK57' in the data.
- For each $i \in I$:
    - $A_i$ = revenue per unit of car model $i$ (parameter, from column 'Revenue')
    - $d_i$ = deterministic demand for car model $i$ (parameter, from column 'Demand')
    - $s_i$ = initial inventory for car model $i$ (parameter, from column 'Initial Inventory')
    - $x_i$ = integer number of units of car model $i$ to fulfill (decision variable)

**Variables**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

**Data Mapping**

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv`
    - Index set $I$: All records where `Product Name` = 'FDK57'
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $s_i$: column `Initial Inventory`
    - Decision variable $x_i$: quantity fulfilled for each $i \in I$