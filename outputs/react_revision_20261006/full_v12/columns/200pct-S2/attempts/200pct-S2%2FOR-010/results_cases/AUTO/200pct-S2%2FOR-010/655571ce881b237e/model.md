### Mathematical Model

Let:
- $S$ = set of sections, indexed by $s$ (from file_0_view_0, column SectionID)
- $P$ = set of products, indexed by $p$ (from file_1_view_0, column ProductName)
- $x_{sp}$ = number of units of product $p$ to stock in section $s$ (decision variable, integer, $\geq 0$)
- $v_p$ = value (revenue) per unit of product $p$ (from file_1_view_0, column Value)
- $w_p$ = space requirement per unit of product $p$ (from file_1_view_0, column Weight)
- $C_s$ = display space capacity of section $s$ (from file_0_view_0, column Capacity)

#### Objective:
Maximize total revenue:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
$$

#### Constraints:
For each section $s \in S$:
$$
\sum_{p \in P} w_p \, x_{sp} \leq C_s
$$

For all $s \in S$, $p \in P$:
$$
x_{sp} \in \mathbb{Z}_{\geq 0}
$$

---

### Data Mapping

- $S$: All SectionID in file_0_view_0 (capacity.csv), column SectionID
- $P$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $v_p$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_p$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $C_s$: file_0_view_0 (capacity.csv), column Capacity, keyed by SectionID
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$

All variables $x_{sp}$ are nonnegative integers. All constraints and parameters are mapped directly from the specified columns and business IDs in the source files.