Mathematical Model

Sets:
I: set of authorized item options (item_ref)  
C: set of categories  
R: set of resources  
B: set of bundle pairs (item_a, item_b)  
IC: set of incompatible pairs (item_a, item_b)  
RQ: set of (item_ref, prerequisite_ref) pairs from requires

Parameters (Data Mapping in table_id.column):
- benefit_i: sum of amount_cents for each item_ref i in file_0_view_0 where table = 'benefit'
- item_fee_i: activation_fee_cents for item_ref i in file_8_view_0 where table = 'item_fee'
- authorized_i: authorized for item_ref i in file_6_view_0 and file_7_view_0 where table = 'item'
- min_lot_i: minimum_lot for item_ref i in file_6_view_0 and file_7_view_0 where table = 'item'
- max_order_i: maximum_order for item_ref i in file_6_view_0 and file_7_view_0 where table = 'item'
- cat_i: category for item_ref i in file_6_view_0 and file_7_view_0 where table = 'item'
- usage_ir: sum of amount for each (item_ref i, resource r) in file_11_view_0 and file_12_view_0 where table = 'usage'
- cap_r: sum of amount for each resource r in file_2_view_0 where table = 'capacity_ledger'
- cat_min_c: minimum_quantity for category c in file_3_view_0 where table = 'category'
- cat_max_c: maximum_quantity for category c in file_3_view_0 where table = 'category'
- cat_fee_c: activation_fee_cents for category c in file_3_view_0 where table = 'category'
- bundle_bonus_b: bonus_cents for bundle b = (item_a, item_b) in file_1_view_0 where table = 'bundle'

Decision variables:
x_i ∈ {0} ∪ {min_lot_i, min_lot_i+1, ..., max_order_i} for i ∈ I (integer, 0 if not selected)
y_i ∈ {0,1} for i ∈ I (1 if x_i > 0, 0 otherwise)
z_c ∈ {0,1} for c ∈ C (1 if any x_i > 0 for i in category c, 0 otherwise)
w_b ∈ {0,1} for b ∈ B (1 if both items in bundle b are selected, 0 otherwise)

Objective:
Maximize
∑_{i∈I} benefit_i x_i
- ∑_{i∈I} item_fee_i y_i
- ∑_{c∈C} cat_fee_c z_c
+ ∑_{b∈B} bundle_bonus_b w_b

Subject to:

1. Authorization and integrality:
x_i = 0 if authorized_i = 0, ∀i∈I  
x_i ∈ {0} ∪ {min_lot_i, ..., max_order_i}, ∀i∈I

2. Activation indicator:
y_i = 1 if x_i > 0, y_i = 0 if x_i = 0, ∀i∈I  
(Enforced by: x_i ≤ max_order_i y_i, x_i ≥ min_lot_i y_i, y_i ∈ {0,1})

3. Category activation:
z_c = 1 if ∑_{i:cat_i=c} x_i > 0, z_c = 0 otherwise, ∀c∈C  
(Enforced by: ∑_{i:cat_i=c} x_i ≤ cat_max_c z_c, ∑_{i:cat_i=c} x_i ≥ cat_min_c z_c or 0)

4. Category quantity limits:
cat_min_c z_c ≤ ∑_{i:cat_i=c} x_i ≤ cat_max_c z_c, ∀c∈C

5. Resource constraints:
∑_{i∈I} usage_ir x_i ≤ cap_r, ∀r∈R

6. Incompatibility:
y_{item_a} + y_{item_b} ≤ 1, ∀(item_a, item_b) ∈ IC

7. Requires:
y_{item_ref} ≤ y_{prerequisite_ref}, ∀(item_ref, prerequisite_ref) ∈ RQ

8. Bundle bonuses:
w_b ≤ y_{item_a}, w_b ≤ y_{item_b}, w_b ≥ y_{item_a} + y_{item_b} - 1, ∀b=(item_a, item_b)∈B

9. Variable domains:
x_i ∈ {0} ∪ {min_lot_i, ..., max_order_i} (integer), y_i ∈ {0,1}, z_c ∈ {0,1}, w_b ∈ {0,1}

Data Mapping:
- file_0_view_0: benefit_i (sum amount_cents by item_ref)
- file_8_view_0: item_fee_i (activation_fee_cents by item_ref)
- file_6_view_0, file_7_view_0: authorized_i, min_lot_i, max_order_i, cat_i (by item_ref)
- file_11_view_0, file_12_view_0: usage_ir (sum amount by item_ref, resource)
- file_2_view_0: cap_r (sum amount by resource)
- file_3_view_0: cat_min_c, cat_max_c, cat_fee_c (by category)
- file_1_view_0: bundle_bonus_b (bonus_cents by (item_a, item_b))
- file_5_view_0: IC (incompatible pairs)
- file_10_view_0: RQ (requires pairs)

All indices, parameters, and constraints are defined directly from the supplied tables and their rows. All units are in USD cents or matching base units as provided. The objective is the net benefit in USD cents.