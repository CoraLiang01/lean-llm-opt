Mathematical Model (Abstract Formulation)

Index Sets:
- 𝒞: Set of components (business IDs: C1, ..., C111 from file_1_view_0["Unnamed: 0"])
- 𝒲: Set of workshops (business IDs: "Casting", "Milling", "Finishing", "Assembly", "QA & Packaging" from file_2_view_0["workshop"])

Parameters:
- p_c: Unit price of component c ∈ 𝒞
  Data Mapping: file_1_view_0["unit_price"], indexed by file_1_view_0["Unnamed: 0"]
- t_{w,c}: Unit processing time required for component c in workshop w
  Data Mapping: file_0_view_0[w, c], with w from file_0_view_0["Unnamed: 0"], c from file_0_view_0 columns C1...C111
- H_w: Total available working hours in workshop w ∈ 𝒲
  Data Mapping: file_2_view_0["total_hours"], indexed by file_2_view_0["workshop"]

Decision Variables:
- x_c: Number of units to produce of component c ∈ 𝒞
  Domain: x_c ∈ ℤ₊ (nonnegative integers)

Objective:
Maximize total output value:
  maximize Z = ∑_{c∈𝒞} p_c · x_c

Constraints:
- Workshop capacity constraints (for each w ∈ 𝒲):
  ∑_{c∈𝒞} t_{w,c} · x_c ≤ H_w

- Nonnegativity and integrality:
  x_c ∈ ℤ₊  ∀ c ∈ 𝒞

Data Mapping Summary:
- 𝒞: All file_1_view_0["Unnamed: 0"] (component business IDs)
- 𝒲: All file_2_view_0["workshop"] (workshop business IDs)
- p_c: file_1_view_0["unit_price"], indexed by file_1_view_0["Unnamed: 0"]
- t_{w,c}: file_0_view_0[w, c], w from file_0_view_0["Unnamed: 0"], c from file_0_view_0 columns C1...C111
- H_w: file_2_view_0["total_hours"], indexed by file_2_view_0["workshop"]

Variables:
- x_c: Number of units to produce of component c (integer, ≥0)

Objective:
- Maximize ∑_{c∈𝒞} p_c · x_c

Constraints:
- For each workshop w ∈ 𝒲: ∑_{c∈𝒞} t_{w,c} · x_c ≤ H_w
- x_c ∈ ℤ₊ ∀ c ∈ 𝒞