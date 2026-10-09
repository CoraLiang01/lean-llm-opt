### Mathematical Model

Let:
- $I$ = set of displays, indexed by $i$ (from all ShelfID in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer)

Parameters:
- $v_j$ = value of product $j$ (Value from file_1_view_0)
- $w_j$ = weight of product $j$ (Weight from file_1_view_0)
- $C_i$ = capacity of display $i$ (Capacity from file_0_view_0)

Let $j^*$ denote the first product in file_1_view_0 (ProductName = "Smartphone").

#### Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

#### Constraints:
1. **Display Capacity Constraints** (for each display $i$):
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq C_i \quad \forall i \in I
   \]
2. **Minimum Quantity of First Product**:
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality**:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: All ShelfID in file_0_view_0["ShelfID"]
- $J$: All ProductName in file_1_view_0["ProductName"]
- $v_j$: file_1_view_0["Value"] (keyed by ProductName)
- $w_j$: file_1_view_0["Weight"] (keyed by ProductName)
- $C_i$: file_0_view_0["Capacity"] (keyed by ShelfID)
- $j^*$: ProductName in file_1_view_0 at source_row 0 ("Smartphone")
- $x_{ij}$: Number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer)