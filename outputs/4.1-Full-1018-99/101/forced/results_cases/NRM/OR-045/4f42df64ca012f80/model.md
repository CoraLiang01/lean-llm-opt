#### Index Sets
- $I$: set of produce types (from file_1_view_0, column ProductName)

#### Parameters
- $v_i$: benefit per unit of produce $i$ (from file_1_view_0, column Value)
- $w_i$: weight per unit of produce $i$ (from file_1_view_0, column Weight)
- $C$: total inventory capacity (from file_0_view_0, column Capacity)

#### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of produce $i$ to order daily

#### Objective
\[
\max \sum_{i \in I} v_i x_i
\]

#### Constraints
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- file_1_view_0 (products.csv): 
    - ProductName $\rightarrow$ $I$
    - Value $\rightarrow$ $v_i$
    - Weight $\rightarrow$ $w_i$
- file_0_view_0 (capacity.csv): 
    - Capacity $\rightarrow$ $C$