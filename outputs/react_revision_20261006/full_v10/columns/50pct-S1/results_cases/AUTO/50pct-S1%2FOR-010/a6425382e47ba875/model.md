Mathematical Model

Index Sets:
- Let S be the set of SectionIDs from file_0_view_0 (capacity.csv).
- Let P be the set of ProductNames from file_1_view_0 (products.csv).

Parameters:
- $c_s$: Capacity of section $s \in S$ (file_0_view_0, column Capacity, key SectionID).
- $v_p$: Value (price) of product $p \in P$ (file_1_view_0, column Value, key ProductName).
- $w_p$: Weight (space requirement) of product $p \in P$ (file_1_view_0, column Weight, key ProductName).

Decision Variables:
- $x_{sp}$: Number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$.

Objective:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
$$

Subject to:
- Section capacity constraints:
$$
\sum_{p \in P} w_p \, x_{sp} \leq c_s \quad \forall s \in S
$$

- Integrality and nonnegativity:
$$
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

Data Mapping

Index Sets:
- S: file_0_view_0, column SectionID
- P: file_1_view_0, column ProductName

Parameters:
- $c_s$: file_0_view_0, column Capacity, key SectionID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

Decision Variables:
- $x_{sp}$: Number of units of product $p$ in section $s$ (indexed by SectionID and ProductName)

All data is mapped directly from the specified columns and keys in the returned CSV files.