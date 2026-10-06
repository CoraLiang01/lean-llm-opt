#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in ZARASales.csv where the "Product Name" contains "FAUX". Each product $i \in I$ is uniquely identified by its "Product Name".

**Parameters:**  
For each $i \in I$ (see Data Mapping below):
- $r_i$ = Revenue per unit of product $i$ ("Revenue" column)
- $d_i$ = Demand for product $i$ ("Demand" column)
- $s_i$ = Initial Inventory of product $i$ ("Initial Inventory" column)

**Decision Variables:**  
For each $i \in I$:
- $x_i$ = Number of units of product $i$ to fulfill  
Domain: $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory and Demand Limits:
   $$
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   $$
   (i.e., cannot fulfill more than available inventory or demand)

2. Integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

| Model Symbol | ZARASales.csv Column | table_id         | Description                                 |
|--------------|---------------------|------------------|---------------------------------------------|
| $i$          | Product Name        | file_0_view_0    | Each product where Product Name contains FAUX|
| $r_i$        | Revenue             | file_0_view_0    | Revenue per unit for product $i$            |
| $d_i$        | Demand              | file_0_view_0    | Demand for product $i$                      |
| $s_i$        | Initial Inventory   | file_0_view_0    | Initial Inventory for product $i$           |

- All parameter values are taken directly from the corresponding columns for each row where "Product Name" contains "FAUX" in ZARASales.csv (table_id: file_0_view_0).
- The decision variable $x_i$ is defined for each such product $i$.

---

**Note:**  
No additional constraints or data are imposed beyond those specified above. All variables, parameters, and constraints are mapped directly from the provided data and user description.