##### Mathematical Model

Let $I$ be the set of all car model records with ‘Product Name’ prefix ‘FDK57’ (see Data Mapping).

**Parameters:**
- $r_i$: Revenue per unit for car model $i$
- $d_i$: Demand for car model $i$
- $s_i$: Initial Inventory for car model $i$

**Decision Variables:**
- $x_i$: Number of units of car model $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory limit for each car model:
$$
x_i \leq s_i \quad \forall i \in I
$$

2. Demand limit for each car model:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0, column ‘Product Name’ with prefix ‘FDK57’
- $r_i$: file_0_view_0, column ‘Revenue’
- $d_i$: file_0_view_0, column ‘Demand’
- $s_i$: file_0_view_0, column ‘Initial Inventory’
- Decision variable $x_i$ is defined for each $i \in I$ (each returned row with ‘FDK57’ prefix in ‘Product Name’)

All parameters are mapped directly from the corresponding columns in table_id file_0_view_0.