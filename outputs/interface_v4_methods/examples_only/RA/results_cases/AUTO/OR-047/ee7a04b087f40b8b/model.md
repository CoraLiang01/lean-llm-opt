ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of platforms (PlatformId from file_0_view_0)
- 𝐺: Set of genres (ProductName from file_1_view_0)

Parameters:
- cap_p: Memory capacity of platform p ∈ 𝑃
  Data Mapping: cap_p = CSVQA_DATA["file_0_view_0"]["Capacity"][PlatformId = p]
- val_g: Value per unit of genre g ∈ 𝐺
  Data Mapping: val_g = CSVQA_DATA["file_1_view_0"]["Value"][ProductName = g]
- w_g: Memory requirement per unit of genre g ∈ 𝐺
  Data Mapping: w_g = CSVQA_DATA["file_1_view_0"]["Weight"][ProductName = g]

Decision Variables:
- x_{p,g}: Number of units of games from genre g to be listed on platform p (integer, x_{p,g} ≥ 0)

Objective:
Maximize total value across all platforms and genres:
\[
\max \sum_{p \in 𝑃} \sum_{g \in 𝐺} val_g \cdot x_{p,g}
\]

Constraints:
1. Platform memory capacity:
   For each platform p ∈ 𝑃,
   \[
   \sum_{g \in 𝐺} w_g \cdot x_{p,g} \leq cap_p
   \]

2. Nonnegativity and integrality:
   For all p ∈ 𝑃, g ∈ 𝐺,
   \[
   x_{p,g} \in \mathbb{Z}_{\geq 0}
   \]

Data Mapping:
- 𝑃 = {PlatformId | rows in CSVQA_DATA["file_0_view_0"]}
- 𝐺 = {ProductName | rows in CSVQA_DATA["file_1_view_0"]}
- cap_p: file_0_view_0, column "Capacity", key "PlatformId"
- val_g: file_1_view_0, column "Value", key "ProductName"
- w_g: file_1_view_0, column "Weight", key "ProductName"

All parameters and sets are mapped directly from the supplied files, preserving row order and explicit business IDs.