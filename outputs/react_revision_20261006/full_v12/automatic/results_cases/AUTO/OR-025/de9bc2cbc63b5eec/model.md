#### Symbolic Mathematical Model

**Index Set:**
- $I$: set of all products where the value in column ‘Product Name’ begins with "TABLET_" in table_id `file_0_view_0`.

**Parameters:**
- $a_i$: revenue per unit of product $i \in I$ (from column ‘Revenue’ in `file_0_view_0`)
- $d_i$: deterministic demand for product $i \in I$ (from column ‘Demand’ in `file_0_view_0`)
- $s_i$: initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in `file_0_view_0`)

**Decision Variables:**
- $x_i$: number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- Index set $I$ and all parameters $a_i$, $d_i$, $s_i$ are defined from rows in table_id `file_0_view_0` of `SmartphoneRetailOutletSalesData.csv` where column ‘Product Name’ starts with "TABLET_".
- $a_i$ is from column ‘Revenue’.
- $d_i$ is from column ‘Demand’.
- $s_i$ is from column ‘Initial Inventory’.