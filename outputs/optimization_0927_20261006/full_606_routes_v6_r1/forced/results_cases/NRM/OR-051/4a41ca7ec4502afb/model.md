#### Index Sets

- $I$: set of cabinets (from file_0_view_0, column CabinetID)
- $J$: set of coffee products (from file_1_view_0, column ProductName)

#### Parameters

- $C_i$: capacity of cabinet $i \in I$ (from file_0_view_0, column Capacity)
- $v_j$: value per unit of product $j \in J$ (from file_1_view_0, column Value)
- $w_j$: weight per unit of product $j \in J$ (from file_1_view_0, column Weight)

#### Decision Variables

- $x_{ij}$: integer number of units of product $j \in J$ to place in cabinet $i \in I$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

- Cabinet capacity constraints:
  $$
  \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
  $$

- Integer nonnegativity:
  $$
  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, \forall j \in J
  $$

---

#### Data Mapping

- file_0_view_0 (capacity.csv): CabinetID $\rightarrow I$, Capacity $\rightarrow C_i$
- file_1_view_0 (products.csv): ProductName $\rightarrow J$, Value $\rightarrow v_j$, Weight $\rightarrow w_j$