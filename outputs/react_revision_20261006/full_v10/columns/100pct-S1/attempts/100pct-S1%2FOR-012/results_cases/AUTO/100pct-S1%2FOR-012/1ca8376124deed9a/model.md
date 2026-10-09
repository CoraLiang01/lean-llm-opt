Mathematical Model

Index Sets:
I: set of platform resource IDs from file_0_view_0.resource_id
J: set of game genres from file_1_view_0.item_name

Parameters:
c_i: memory capacity of platform i, from file_0_view_0.resource_capacity
v_j: value per unit of genre j, from file_1_view_0.item_value
a_j: memory requirement per unit of genre j, from file_1_view_0.resource_requirement

Decision Variables:
x_{ij}: number of units of games from genre j to be listed on platform i; integer, x_{ij} ≥ 0

Objective:
maximize  ∑_{i∈I} ∑_{j∈J} v_j x_{ij}

Subject to:
∑_{j∈J} a_j x_{ij} ≤ c_i  ∀ i ∈ I

x_{ij} ∈ ℤ_{≥0}  ∀ i ∈ I, j ∈ J

Data Mapping

Index Sets:
I = file_0_view_0.resource_id
J = file_1_view_0.item_name

Parameters:
c_i = file_0_view_0.resource_capacity, keyed by resource_id
v_j = file_1_view_0.item_value, keyed by item_name
a_j = file_1_view_0.resource_requirement, keyed by item_name

Decision Variables:
x_{ij}: number of units of games from genre j (file_1_view_0.item_name) to be listed on platform i (file_0_view_0.resource_id)

Objective:
maximize total value using v_j from file_1_view_0.item_value

Constraints:
For each i ∈ file_0_view_0.resource_id:
  ∑_{j∈file_1_view_0.item_name} file_1_view_0.resource_requirement[j] × x_{ij} ≤ file_0_view_0.resource_capacity[i]

Variable domains:
x_{ij} ∈ ℤ_{≥0} for all i, j