Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of Bookshelf IDs (from file_0_view_0.BookshelfID)
- 𝑱: Set of Product Names (from file_1_view_0.ProductName)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of bookshelf i ∈ 𝑰 (from file_0_view_0.Capacity)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of book j ∈ 𝑱 (from file_1_view_0.Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight of book j ∈ 𝑱 (from file_1_view_0.Weight)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of book j ∈ 𝑱 to place on bookshelf i ∈ 𝑰 (integer, 𝑥ᵢⱼ ≥ 0)

Objective:
Maximize total value of books placed:
\[
\max \sum_{i \in 𝑰} \sum_{j \in 𝑱} 𝑉𝑎𝑙𝑢𝑒ⱼ \cdot 𝑥ᵢⱼ
\]

Subject to:
- Capacity constraints for each bookshelf:
\[
\sum_{j \in 𝑱} 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ \cdot 𝑥ᵢⱼ \leq 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ \quad \forall i \in 𝑰
\]
- Nonnegativity and integrality:
\[
𝑥ᵢⱼ \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰,\, j \in 𝑱
\]

Data Mapping

Index Sets:
- 𝑰: All BookshelfID values from file_0_view_0.BookshelfID
- 𝑱: All ProductName values from file_1_view_0.ProductName

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0.Capacity, indexed by file_0_view_0.BookshelfID
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0.Value, indexed by file_1_view_0.ProductName
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0.Weight, indexed by file_1_view_0.ProductName

Decision Variables:
- 𝑥ᵢⱼ: Integer, ≥ 0, indexed by (file_0_view_0.BookshelfID, file_1_view_0.ProductName)