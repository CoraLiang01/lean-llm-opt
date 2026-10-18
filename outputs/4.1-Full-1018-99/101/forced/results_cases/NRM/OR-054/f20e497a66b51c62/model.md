#### Index Sets
- $\mathcal{I}$: set of shelves (from file_0_view_0, column ShelfID)
- $\mathcal{J}$: set of products (from file_1_view_0, column ProductName)

#### Parameters
- $C_i$: capacity of shelf $i \in \mathcal{I}$ (from file_0_view_0, column Capacity)
- $v_j$: value per unit of product $j \in \mathcal{J}$ (from file_1_view_0, column Value)
- $w_j$: weight per unit of product $j \in \mathcal{J}$ (from file_1_view_0, column Weight)

#### Decision Variables
- $x_{ij} \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I},\ j \in \mathcal{J}$  
  (number of units of product $j$ placed on shelf $i$)

#### Objective
$$
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} v_j \, x_{ij}
$$

#### Constraints

1. **Shelf Capacity Constraints**  
   For all $i \in \mathcal{I}$:
   $$
   \sum_{j \in \mathcal{J}} w_j \, x_{ij} \leq C_i
   $$

2. **Nonnegativity and Integrality**  
   $$
   x_{ij} \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I},\ j \in \mathcal{J}
   $$

---

#### Data Mapping

- **file_0_view_0 (capacity.csv):**
  - Shelf index set $\mathcal{I}$: column ShelfID
  - Shelf capacity $C_i$: column Capacity

- **file_1_view_0 (products.csv):**
  - Product index set $\mathcal{J}$: column ProductName
  - Product value $v_j$: column Value
  - Product weight $w_j$: column Weight

All records from both tables are included as per FALLBACK_FULL_DATA.