ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of bookshelves (BookshelfID from file_0_view_0 in capacity.csv)
- 𝑱: Set of books (ProductName from file_1_view_0 in products.csv)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of bookshelf i ∈ 𝑰  
  Data Mapping: file_0_view_0, column "Capacity", indexed by "BookshelfID"
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of book j ∈ 𝑱  
  Data Mapping: file_1_view_0, column "Value", indexed by "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight of book j ∈ 𝑱  
  Data Mapping: file_1_view_0, column "Weight", indexed by "ProductName"

Decision Variables:
- 𝑥ᵢⱼ: Number of units of book j ∈ 𝑱 to place on bookshelf i ∈ 𝑰  
  Domain: 𝑥ᵢⱼ ∈ ℤ₊ (nonnegative integers)

Objective:
Maximize total value of books placed on all bookshelves:
\[
\max \sum_{i \in 𝑰} \sum_{j \in 𝑱} 𝑉𝑎𝑙𝑢𝑒ⱼ \cdot 𝑥ᵢⱼ
\]

Constraints:
1. Capacity constraint for each bookshelf:
\[
\forall i \in 𝑰: \quad \sum_{j \in 𝑱} 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ \cdot 𝑥ᵢⱼ \leq 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ
\]
2. Nonnegativity and integrality:
\[
\forall i \in 𝑰, \forall j \in 𝑱: \quad 𝑥ᵢⱼ \in \mathbb{Z}_+, \quad 𝑥ᵢⱼ \geq 0
\]

Data Mapping:
- 𝑰 (BookshelfID): file_0_view_0, column "BookshelfID"
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column "Capacity", indexed by "BookshelfID"
- 𝑱 (ProductName): file_1_view_0, column "ProductName"
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column "Value", indexed by "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column "Weight", indexed by "ProductName"

All bookshelf and book records are included in the index sets, and all parameters are mapped directly from the supplied files as described above.