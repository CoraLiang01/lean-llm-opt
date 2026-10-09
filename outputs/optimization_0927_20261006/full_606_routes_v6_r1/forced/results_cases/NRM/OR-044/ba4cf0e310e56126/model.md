#### Index Sets

- $\mathcal{I}$: Set of sections (from file_0_view_0, column SectionID)
- $\mathcal{J}$: Set of products (from file_1_view_0, column ProductName)

#### Parameters

- $C_i$: Capacity of section $i \in \mathcal{I}$ (from file_0_view_0, column Capacity)
- $v_j$: Value (price) of product $j \in \mathcal{J}$ (from file_1_view_0, column Value)
- $w_j$: Shelf space requirement (weight) of product $j \in \mathcal{J}$ (from file_1_view_0, column Weight)

#### Decision Variables

- $x_{ij}$: Number of units of product $j$ to be placed in section $i$, $\forall i \in \mathcal{I}, j \in \mathcal{J}$

#### Variable Domains

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in \mathcal{I}, j \in \mathcal{J}$

#### Objective Function

$$
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} v_j \cdot x_{ij}
$$

#### Constraints

1. **Section Capacity Constraints** (for all $i \in \mathcal{I}$):

$$
\sum_{j \in \mathcal{J}} w_j \cdot x_{ij} \leq C_i
$$

2. **Non-negativity and Integrality** (for all $i \in \mathcal{I}, j \in \mathcal{J}$):

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

#### Data Mapping

- $\mathcal{I}$ (sections): file_0_view_0, column SectionID
- $C_i$ (section capacities): file_0_view_0, column Capacity
- $\mathcal{J}$ (products): file_1_view_0, column ProductName
- $v_j$ (product values): file_1_view_0, column Value
- $w_j$ (product weights): file_1_view_0, column Weight

All data is used as returned by the query, with no additional filtering or transformation.