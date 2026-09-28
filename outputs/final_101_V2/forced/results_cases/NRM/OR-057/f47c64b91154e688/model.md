### Abstract Mathematical Model

#### Index Sets
- $I$: set of platforms (from capacity.csv, column PlatformID)
- $J$: set of games (from products.csv, column ProductName)

#### Parameters
- $C_i$: memory capacity of platform $i \in I$ (from capacity.csv, column Capacity)
- $v_j$: value of game $j \in J$ (from products.csv, column Value)
- $w_j$: memory requirement of game $j \in J$ (from products.csv, column Weight)

#### Decision Variables
- $x_{ij}$: integer number of units of game $j$ to be listed on platform $i$, $\forall i \in I, j \in J$, with $x_{ij} \in \mathbb{Z}_{\geq 0}$

#### Objective
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Platform Memory Capacity Constraints:**
   $$
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   $$

2. **Variable Domain Constraints:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, j \in J
   $$

---

### Data Mapping

- **capacity.csv**
  - Table ID: file_0_view_0
    - PlatformID $\rightarrow$ $I$
    - Capacity $\rightarrow$ $C_i$
- **products.csv**
  - Table ID: file_1_view_0
    - ProductName $\rightarrow$ $J$
    - Value $\rightarrow$ $v_j$
    - Weight $\rightarrow$ $w_j$

No literal record values or counts are included in the model. All index sets and parameters are defined symbolically and mapped to their exact source columns.