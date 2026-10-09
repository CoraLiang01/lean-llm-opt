Mathematical Model

Index Sets:
I: set of vehicle types (ProductName from file_1_view_0)

Parameters:
v_i: profit per unit of vehicle i (Value from file_1_view_0, for each i ∈ I)
w_i: inventory weight per unit of vehicle i (Weight from file_1_view_0, for each i ∈ I)
C: total inventory capacity (Capacity from file_0_view_0)

Decision Variables:
x_i: number of vehicles of type i to order per day (integer, x_i ≥ 0, for each i ∈ I)

Objective:
maximize ∑_{i∈I} v_i x_i

Subject to:
  ∑_{i∈I} w_i x_i ≤ C
  x_i ∈ ℤ_≥0  ∀ i ∈ I

Data Mapping

Index Sets:
I: file_1_view_0.ProductName

Parameters:
v_i: file_1_view_0.Value, keyed by ProductName
w_i: file_1_view_0.Weight, keyed by ProductName
C: file_0_view_0.Capacity

Decision Variables:
x_i: number of vehicles of type i to order per day, indexed by file_1_view_0.ProductName

Objective:
maximize ∑_{i∈I} v_i x_i

Constraints:
  ∑_{i∈I} w_i x_i ≤ C
  x_i ∈ ℤ_≥0  ∀ i ∈ I