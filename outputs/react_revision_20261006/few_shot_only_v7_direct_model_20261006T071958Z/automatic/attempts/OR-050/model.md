ABSTRACT MATHEMATICAL MODEL

**Index Sets:**
- $I$: set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$: set of products (indexed by $j$), corresponding to all ProductName in file_1_view_0.

**Parameters:**
- $c_i$: capacity of display $i$ (from Capacity in file_0_view_0).
- $v_j$: value per unit of product $j$ (from Value in file_1_view_0).
- $w_j$: weight per unit of product $j$ (from Weight in file_1_view_0).

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**  
   For each display $i \in I$,
   \[
   \sum_{j \in J} w_j x_{ij} \leq c_i
   \]
2. **Minimum Quantity for First Product:**  
   Let $j^*$ denote the index of the first product in file_1_view_0 (i.e., the product in the first record, which is "Smartphone"):
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$: All ShelfID in file_0_view_0 (/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv, column "ShelfID")
- $J$: All ProductName in file_1_view_0 (/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv, column "ProductName")
- $c_i$: file_0_view_0, column "Capacity", keyed by "ShelfID"
- $v_j$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_j$: file_1_view_0, column "Weight", keyed by "ProductName"
- $j^*$: ProductName in the first record of file_1_view_0 ("Smartphone")

All index sets, parameters, and constraints are mapped directly to the supplied data.