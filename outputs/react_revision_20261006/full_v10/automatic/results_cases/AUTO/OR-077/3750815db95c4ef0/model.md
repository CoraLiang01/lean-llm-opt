Mathematical Model

Sets:
I = {1, 2, ..., 140}   (items, from value.csv, column "item")

Parameters:
v_i = value of item i (from value.csv, column "value", table_id: file_0_view_0)
w_i = weight of item i (from value.csv, column "weight", table_id: file_0_view_0)
W = 15   (total weight limit)

Decision Variables:
x_i ∈ {0,1}   for all i ∈ I   (1 if item i is selected, 0 otherwise)

Objective:
maximize   ∑_{i∈I} v_i x_i

Subject to:
∑_{i∈I} w_i x_i ≤ W
x_i ∈ {0,1}   for all i ∈ I

Data Mapping:
Set I, and parameters v_i, w_i are defined by the rows and columns "item", "value", "weight" in value.csv (table_id: file_0_view_0). The weight limit W = 15 is from the user description.