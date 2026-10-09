Symbolic Mathematical Model

Sets:
I = set of authorized item_refs (from file_5_view_0)
C = set of categories (from file_2_view_0)
R = set of resources (from file_1_view_0)
B = set of bundle bonus pairs (from file_0_view_0)
P = set of incompatible pairs (from file_4_view_0)
Q = set of prerequisite pairs (from file_7_view_0)

Parameters:
unit_benefit_cents[i] = unit benefit for item i ∈ I (file_5_view_0)
item_fee_cents[i] = fixed fee for item i ∈ I (file_5_view_0)
category[i] = category of item i ∈ I (file_5_view_0)
minimum_lot[i], maximum_order[i] = min/max order for item i ∈ I (file_5_view_0)
minimum_quantity[c], maximum_quantity[c] = min/max total for category c ∈ C (file_2_view_0)
activation_fee_cents[c] = activation fee for category c ∈ C (file_2_view_0)
amount[i,r], unit[i,r] = per-unit usage and unit for item i ∈ I, resource r ∈ R (file_8_view_0)
capacity_ledger[r] = sum of all 'amount' for resource r (file_1_view_0, sum over all entries for r)
bonus_cents[b] = bonus for bundle b ∈ B (file_0_view_0)
bundle_items[b] = (item_a, item_b) for bundle b ∈ B (file_0_view_0)
incompatible_pairs = set of (item_a, item_b) ∈ P (file_4_view_0)
prerequisite_pairs = set of (item_ref, prerequisite_ref) ∈ Q (file_7_view_0)

Decision Variables:
x[i] ∈ ℤ₊, ∀i ∈ I (number of cases of item i ordered)
y[i] ∈ {0,1}, ∀i ∈ I (1 if x[i] ≥ 1, 0 otherwise)
z[c] ∈ {0,1}, ∀c ∈ C (1 if any item in category c is ordered, 0 otherwise)
w[b] ∈ {0,1}, ∀b ∈ B (1 if both items in bundle b are ordered, 0 otherwise)

Model:

Maximize net benefit in USD cents:
max
  ∑_{i∈I} unit_benefit_cents[i] * x[i]
- ∑_{i∈I} item_fee_cents[i] * y[i]
- ∑_{c∈C} activation_fee_cents[c] * z[c]
+ ∑_{b∈B} bonus_cents[b] * w[b]

Subject to:

// Item order bounds
∀i ∈ I:
    x[i] = 0  or  minimum_lot[i] ≤ x[i] ≤ maximum_order[i]

// Link y[i] to x[i]
∀i ∈ I:
    y[i] ≥ 1 if x[i] ≥ 1; y[i] = 0 if x[i] = 0
    (i.e., y[i] ∈ {0,1}, y[i] ≥ x[i]/maximum_order[i])

// Category totals and activation
∀c ∈ C:
    minimum_quantity[c] ≤ ∑_{i∈I: category[i]=c} x[i] ≤ maximum_quantity[c]
    z[c] ≥ y[i] for all i ∈ I with category[i]=c

// Resource constraints (unit conversions applied)
For each r ∈ R:
    ∑_{i∈I} (amount[i,r] * x[i] * unit_conversion[i,r]) ≤ capacity_ledger[r]
where unit_conversion[i,r] is:
    - 1000 if unit[i,r] = 'liter', capacity_ledger[r] in 'ml'
    - 60 if unit[i,r] = 'hour', capacity_ledger[r] in 'minute'
    - 1000 if unit[i,r] = 'kwh', capacity_ledger[r] in 'wh'
    - 1 otherwise

// Incompatibility
∀(i,j) ∈ incompatible_pairs:
    y[i] + y[j] ≤ 1

// Prerequisite
∀(i,pr) ∈ prerequisite_pairs:
    y[i] ≤ y[pr]

// Bundle bonuses
∀b ∈ B, let (i,j) = bundle_items[b]:
    w[b] ≤ y[i]
    w[b] ≤ y[j]
    w[b] ≥ y[i] + y[j] - 1

// Integrality
∀i ∈ I: x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] ∩ ℤ
∀i ∈ I: y[i] ∈ {0,1}
∀c ∈ C: z[c] ∈ {0,1}
∀b ∈ B: w[b] ∈ {0,1}

Data Mapping:
- file_0_view_0: bundle bonus (B, bonus_cents, item_a, item_b)
- file_1_view_0: resource capacity ledger (R, capacity_ledger)
- file_2_view_0: category constraints (C, minimum_quantity, maximum_quantity, activation_fee_cents)
- file_3_view_0: item identity (item_ref, entity_id)
- file_4_view_0: incompatible pairs (P)
- file_5_view_0: item order options (I, category, minimum_lot, maximum_order, unit_benefit_cents, item_fee_cents)
- file_7_view_0: item prerequisites (Q)
- file_8_view_0: item resource usage (amount, unit for each i,r)

All sets, parameters, and constraints are defined directly from the supplied tables and their rows. All units are converted as specified. Only authorized=1 items are included. All constraints and bonuses are enforced as described. The objective is the maximum net benefit in USD cents.