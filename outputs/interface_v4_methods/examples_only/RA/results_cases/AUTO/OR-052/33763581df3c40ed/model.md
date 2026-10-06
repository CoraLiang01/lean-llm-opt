ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of bookshelves (BookshelfID from file_0_view_0)
- 𝐵: Set of books (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ₛ: Capacity of bookshelf s ∈ 𝑆  
  Data Mapping: file_0_view_0, column "Capacity", indexed by "BookshelfID"
- 𝑉𝑎𝑙𝑢𝑒_b: Value of book b ∈ 𝐵  
  Data Mapping: file_1_view_0, column "Value", indexed by "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡_b: Weight of book b ∈ 𝐵  
  Data Mapping: file_1_view_0, column "Weight", indexed by "ProductName"

Decision Variables:
- 𝑥ₛb: Number of units of book b ∈ 𝐵 to place on bookshelf s ∈ 𝑆  
  Domain: Integer, 𝑥ₛb ≥ 0

Objective:
Maximize total value of books placed:
\[
\max \sum_{s \in S} \sum_{b \in B} \text{Value}_b \cdot x_{sb}
\]

Constraints:
1. Shelf capacity constraints (for each bookshelf s ∈ 𝑆):
\[
\sum_{b \in B} \text{Weight}_b \cdot x_{sb} \leq \text{Capacity}_s
\]

2. Nonnegativity and integrality:
\[
x_{sb} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, b \in B
\]

DATA MAPPING

- Bookshelf set S and parameter Capacityₛ: file_0_view_0, columns "BookshelfID", "Capacity"
- Book set B and parameters Value_b, Weight_b: file_1_view_0, columns "ProductName", "Value", "Weight"
- Decision variables xₛb are indexed by ("BookshelfID", "ProductName") pairs.

No additional constraints or data are imposed. All bookshelf and book records are included as per the original row order.