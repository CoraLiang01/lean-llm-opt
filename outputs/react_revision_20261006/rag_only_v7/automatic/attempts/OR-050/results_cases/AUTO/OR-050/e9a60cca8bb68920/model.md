Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑆: Set of displays (shelves), indexed by i. (from file_0_view_0, column ShelfID)
- 𝑃: Set of products, indexed by j. (from file_1_view_0, column ProductName)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity (maximum total weight) of display i. (from file_0_view_0, column Capacity)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of product j. (from file_1_view_0, column Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight per unit of product j. (from file_1_view_0, column Weight)
- 𝑗₁: The first product in file_1_view_0 (ProductName at source_row 0).

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j placed on display i. (integer, ≥ 0)

Objective:
Maximize total value of all products placed:
\[
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
\]

Constraints:
1. Display capacity (weight) constraints:
\[
\forall i \in S: \quad \sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]

2. Minimum total quantity of the first product across all displays:
\[
\sum_{i \in S} x_{i j_1} \geq 5
\]

3. Nonnegativity and integrality:
\[
\forall i \in S, \forall j \in P: \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Data Mapping

Index Sets:
- 𝑆: All ShelfID values from file_0_view_0, column ShelfID.
- 𝑃: All ProductName values from file_1_view_0, column ProductName.

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column Capacity, keyed by ShelfID.
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column Value, keyed by ProductName.
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column Weight, keyed by ProductName.
- 𝑗₁: ProductName at source_row 0 in file_1_view_0.

Decision Variables:
- 𝑥ᵢⱼ: For each (ShelfID, ProductName) pair.

All sets, parameters, and constraints are defined using the exact columns and row order as returned in the CSVQA data.