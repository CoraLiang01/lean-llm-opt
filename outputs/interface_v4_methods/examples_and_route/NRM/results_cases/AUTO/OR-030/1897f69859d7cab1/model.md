#### Abstract Mathematical Model

Let:
- $I$ = set of car models with ‘Product Name’ prefix ‘FDK57’ (indexed by $i$; see Data Mapping).
- $r_i$ = revenue per unit for car model $i$ (parameter).
- $d_i$ = demand for car model $i$ (parameter).
- $s_i$ = initial inventory for car model $i$ (parameter).
- $x_i$ = number of units of car model $i$ to fulfill (decision variable, nonnegative integer).

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory limit for each car model:
$$
x_i \leq s_i \qquad \forall i \in I
$$

2. Demand limit for each car model:
$$
x_i \leq d_i \qquad \forall i \in I
$$

3. Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

---

#### Data Mapping

- Index set $I$ and all parameters are defined by the following table and columns:

| table_id           | column name         | symbol | description                        |
|--------------------|--------------------|--------|------------------------------------|
| file_0_view_0      | Product Name       | $i$    | Car model (prefix ‘FDK57’)         |
| file_0_view_0      | Revenue            | $r_i$  | Revenue per unit for model $i$     |
| file_0_view_0      | Demand             | $d_i$  | Demand for model $i$               |
| file_0_view_0      | Initial Inventory  | $s_i$  | Initial inventory for model $i$    |

- Each row in file_0_view_0 corresponds to a distinct car model $i$ (with ‘Product Name’ = ‘FDK57’).

---

**All parameters and variables are indexed by the original table row and column identifiers.**