Mathematical Model for Maximum Net Benefit Vehicle Inventory Replenishment

Sets:
I: set of items (from item table, after selection rules)
C: set of categories (from category table, after selection rules)
R: set of resources (from usage and capacity_ledger tables, after selection rules)
O: set of options (items eligible for order: authorized=1)
B: set of bundle pairs (from bundle table, after selection rules)
P: set of incompatible pairs (from incompatible table, after selection rules)
Q: set of requires pairs (from requires table, after selection rules)

Parameters:
benefit_i: per-unit net benefit for item i ∈ I (sum of amount_cents from benefit table, after selection rules)
item_fee_i: fixed fee for item i ∈ I (from item_fee table, after selection rules)
min_lot_i: minimum lot size for item i ∈ I (from item table)
max_order_i: maximum order for item i ∈ I (from item table)
authorized_i: 1 if item i is authorized, 0 otherwise (from item table)
cat_i: category of item i (from item table)
min_cat_c, max_cat_c: min/max quantity for category c ∈ C (from category table)
cat_fee_c: activation fee for category c ∈ C (from category table)
usage_ir: resource r usage per unit of item i (from usage table, after selection rules; convert units as needed)
cap_r: available capacity for resource r ∈ R (sum of amount from capacity_ledger table, after selection rules; convert units as needed)
bundle_bonus_{ij}: bonus for bundle (i,j) ∈ B (from bundle table)
incompat_{ij}: 1 if (i,j) ∈ P, 0 otherwise (from incompatible table)
requires_{ij}: 1 if (i,j) ∈ Q, 0 otherwise (from requires table)

Decision Variables:
x_i ∈ ℤ_+ : quantity to order of item i ∈ I
y_i ∈ {0,1} : 1 if x_i > 0, 0 otherwise (option activation)
z_c ∈ {0,1} : 1 if any item in category c ∈ C is ordered (category activation)
w_{ij} ∈ {0,1} : 1 if both x_i > 0 and x_j > 0 for bundle (i,j) ∈ B (bundle awarded)

Objective:
Maximize net benefit in USD cents:
max
∑_{i∈I} benefit_i x_i
− ∑_{i∈I} item_fee_i y_i
− ∑_{c∈C} cat_fee_c z_c
+ ∑_{(i,j)∈B} bundle_bonus_{ij} w_{ij}

Subject to:

// Option authorization and lot constraints
x_i = 0, if authorized_i = 0, ∀i∈I
min_lot_i y_i ≤ x_i ≤ max_order_i y_i, ∀i∈I
y_i ∈ {0,1}, x_i ∈ ℤ_+, ∀i∈I

// Category activation and quantity limits
z_c ≥ y_i, ∀i∈I: cat_i = c
min_cat_c z_c ≤ ∑_{i:cat_i=c} x_i ≤ max_cat_c z_c, ∀c∈C
z_c ∈ {0,1}, ∀c∈C

// Resource capacity constraints (with unit conversions)
∑_{i∈I} usage_ir x_i ≤ cap_r, ∀r∈R

// Incompatibility constraints
y_i + y_j ≤ 1, ∀(i,j)∈P

// Requires constraints
y_i ≤ y_j, ∀(i,j)∈Q

// Bundle bonus activation
w_{ij} ≤ y_i, w_{ij} ≤ y_j, w_{ij} ≥ y_i + y_j − 1, ∀(i,j)∈B
w_{ij} ∈ {0,1}, ∀(i,j)∈B

// Variable domains
x_i ∈ ℤ_+, y_i ∈ {0,1}, z_c ∈ {0,1}, w_{ij} ∈ {0,1}

Data Mapping:
- All sets, parameters, and variables are defined from the current CSV data after applying the selection rules:
  - For each (tenant, table, record_id), keep the highest integer revision on or before 2026-03-12, discard if DELETE, deduplicate retransmissions.
  - Use only records for tenant = NORTH.
  - For each table, apply these rules before joining or summing.
- benefit_i: sum of amount_cents for each item_ref in benefit table (file_4_view_0).
- item_fee_i: activation_fee_cents for each item_ref in item_fee table (file_2_view_0).
- min_lot_i, max_order_i, authorized_i, cat_i: from item table (file_9_view_0).
- min_cat_c, max_cat_c, cat_fee_c: from category table (file_15_view_0).
- usage_ir: amount for each (item_ref, resource) in usage table (file_5_view_0), converted to the units of cap_r.
- cap_r: sum of amount for each resource in capacity_ledger table (file_7_view_0), using the latest revision.
- bundle_bonus_{ij}: bonus_cents for each (item_a, item_b) in bundle table (file_3_view_0).
- incompat_{ij}: from incompatible table (file_6_view_0).
- requires_{ij}: from requires table (file_0_view_0).

Unit conversions:
- 1000 ml = 1 liter
- 60 minutes = 1 hour
- 1000 wh = 1 kwh
- All resource usage and capacity are converted to the same units before comparison.

All constraints, sets, and parameters are derived from the current data as described above. The model maximizes net benefit in USD cents, deducting all fixed fees and awarding bundle bonuses as specified.