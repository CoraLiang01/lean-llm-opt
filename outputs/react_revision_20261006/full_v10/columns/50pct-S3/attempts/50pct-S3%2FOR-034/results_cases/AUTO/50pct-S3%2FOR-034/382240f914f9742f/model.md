Mathematical Model

Sets:
I = {1, 2, ..., 140}   (items, from value.csv column "item")

Parameters (from value.csv, table_id: file_0_view_0):
v_i = value of item i          (column "value")
w_i = weight of item i         (column "weight")
W = 15                        (total weight limit)

Decision Variables:
x_i ∈ {0,1}   for i ∈ I       (1 if item i is selected, 0 otherwise)

Objective:
maximize   ∑_{i∈I} v_i x_i

Subject to:
∑_{i∈I} w_i x_i ≤ W
x_i ∈ {0,1}   for all i ∈ I

Data Mapping:
Set I, and parameters v_i, w_i are defined by all rows of value.csv (table_id: file_0_view_0), columns "item", "value", "weight". The total weight limit W = 15 is from the user description.