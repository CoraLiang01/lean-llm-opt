**Mathematical Model**

Let:
- $I$ = set of all products with SKU starting with 'ZZ' (i.e., all 'ZZ' products in the current dataset).
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column 'Revenue')
    - $d_i$ = demand for product $i$ (from column 'Demand')
    - $s_i$ = initial inventory for product $i$ (from column 'Initial Inventory')
    - $x_i$ = integer decision variable: number of units of product $i$ to fulfill

**Variables**
- $x_i \in \mathbb{Z}_+$, $0 \leq x_i \leq \min\{d_i, s_i\}$, for all $i \in I$

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

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv`
    - Index set $I$: all records where column `SKU` starts with 'ZZ'
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $s_i$: column `Initial Inventory`
    - Variable $x_i$: decision variable for each $i \in I$