Sets:
I = {1, 2, ..., 140}   (items, from value.csv column "item")

Parameters (from value.csv, table_id: file_0_view_0):
v_i = value of item i      (column "value")
w_i = weight of item i     (column "weight")
W = 15                    (merchandise counter weight limit)

Decision variables:
x_i ∈ {0,1}   for i ∈ I   (1 if item i is selected, 0 otherwise)

Objective:
maximize   ∑_{i∈I} v_i x_i

Subject to:
∑_{i∈I} w_i x_i ≤ W
x_i ∈ {0,1}   for all i ∈ I

Data Mapping:
I: file_0_view_0, column "item"
v_i: file_0_view_0, column "value"
w_i: file_0_view_0, column "weight"
W: user description (15)