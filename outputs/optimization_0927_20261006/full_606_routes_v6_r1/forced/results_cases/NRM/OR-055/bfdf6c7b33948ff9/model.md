#### Index Sets

- $I$: set of display areas (from `capacity.csv`, column `DisplayID`)
- $J$: set of boat types (from `products.csv`, column `ProductName`)

#### Parameters

- $C_i$: capacity of display area $i \in I$ (from `capacity.csv`, column `Capacity`)
- $v_j$: value of one unit of boat type $j \in J$ (from `products.csv`, column `Value`)
- $w_j$: size of one unit of boat type $j \in J$ (from `products.csv`, column `Weight`)

#### Decision Variables

- $x_{ij}$: number of units of boat type $j \in J$ to place in display area $i \in I$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Capacity constraints for each display area:**
   $$
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   $$

2. **Non-negativity and integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, \forall j \in J
   $$

---

#### Data Mapping

- $I$ (display areas), $C_i$: from `capacity.csv`, columns `DisplayID`, `Capacity`, table_id: `file_0_view_0`
- $J$ (boat types), $v_j$, $w_j$: from `products.csv`, columns `ProductName`, `Value`, `Weight`, table_id: `file_1_view_0`