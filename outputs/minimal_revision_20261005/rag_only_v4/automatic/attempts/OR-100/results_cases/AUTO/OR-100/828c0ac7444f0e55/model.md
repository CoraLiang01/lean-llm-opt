ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝒞: Set of components (business IDs from file_1_view_0['Unnamed: 0'])
- 𝒲: Set of workshops (business IDs from file_2_view_0['workshop'])

Parameters:
- p_c: Unit price of component c ∈ 𝒞  
  (Data: file_1_view_0['unit_price'], indexed by file_1_view_0['Unnamed: 0'])
- t_{w,c}: Unit processing time required for component c in workshop w  
  (Data: file_0_view_0[w, c], with workshop w = file_0_view_0['Unnamed: 0'], component c = column name)
- H_w: Total available working hours in workshop w ∈ 𝒲  
  (Data: file_2_view_0['total_hours'], indexed by file_2_view_0['workshop'])

Decision Variables:
- x_c: Number of units to produce of component c ∈ 𝒞  
  Domain: x_c ∈ ℤ₊ (nonnegative integers)

Objective:
Maximize total output value:
\[
\max \sum_{c \in 𝒞} p_c \cdot x_c
\]

Constraints:
- Workshop capacity constraints (for each w ∈ 𝒲):
\[
\sum_{c \in 𝒞} t_{w,c} \cdot x_c \leq H_w
\]

- Nonnegativity and integrality:
\[
x_c \in \mathbb{Z}_+, \quad \forall c \in 𝒞
\]

DATA MAPPING

- 𝒞: All values in file_1_view_0['Unnamed: 0']
- 𝒲: All values in file_2_view_0['workshop']
- p_c: file_1_view_0['unit_price'], indexed by file_1_view_0['Unnamed: 0']
- t_{w,c}: file_0_view_0[w, c], with w = file_0_view_0['Unnamed: 0'], c = column name (C1, ..., C111)
- H_w: file_2_view_0['total_hours'], indexed by file_2_view_0['workshop']
- x_c: Decision variable for each c ∈ 𝒞

All index sets, parameters, and constraints are mapped directly to the supplied data tables and columns. No data is invented or omitted.