ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑰: Set of display areas (indexed by i), corresponding to DisplayID in file_0_view_0.
- 𝑱: Set of vessel types (indexed by j), corresponding to ProductName in file_1_view_0.

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of display area i. [from file_0_view_0, column Capacity]
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of vessel type j. [from file_1_view_0, column Value]
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Size (weight) of vessel type j. [from file_1_view_0, column Weight]

Decision Variables:
- 𝑥ᵢⱼ: Number of vessels of type j to place in display area i. (integer, 𝑥ᵢⱼ ≥ 0)

Objective:
Maximize total value of vessels displayed:
\[
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \text{Value}_j \cdot x_{ij}
\]

Subject to:

1. Capacity constraints for each display area:
\[
\sum_{j \in \mathcal{J}} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in \mathcal{I}
\]

2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I},\ j \in \mathcal{J}
\]

---

DATA MAPPING

- 𝑰 (Display areas): All DisplayID values from file_0_view_0 (capacity.csv), column DisplayID.
- 𝑱 (Vessel types): All ProductName values from file_1_view_0 (products.csv), column ProductName.
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, columns DisplayID (for i), Capacity.
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, columns ProductName (for j), Value.
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, columns ProductName (for j), Weight.
- 𝑥ᵢⱼ: Decision variable for each (DisplayID i, ProductName j) pair.