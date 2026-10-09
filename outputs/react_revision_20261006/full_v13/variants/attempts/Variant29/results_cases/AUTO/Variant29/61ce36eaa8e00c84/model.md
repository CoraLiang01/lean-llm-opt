## Symbolic Mathematical Model

**Sets**
- $I$: set of authorized item options (from file_2_view_0, column item_ref)
- $C$: set of categories (from file_4_view_0, column category)
- $R$: set of resources (from file_8_view_0, column resource)
- $B$: set of bundle pairs (from file_7_view_0, columns item_a, item_b)
- $P$: set of prerequisite pairs (from file_0_view_0, columns item_ref, prerequisite_ref)
- $Q$: set of incompatible pairs (from file_5_view_0, columns item_a, item_b)

**Parameters**
- $u_i$: unit benefit in cents for item $i$ (file_2_view_0, unit_benefit_cents)
- $f_i$: fixed item fee in cents for item $i$ (file_2_view_0, item_fee_cents)
- $cat(i)$: category of item $i$ (file_2_view_0, category)
- $min_i$: minimum lot for item $i$ (file_2_view_0, minimum_lot)
- $max_i$: maximum order for item $i$ (file_2_view_0, maximum_order)
- $a_{i,r}$: per-unit usage of resource $r$ by item $i$ (file_3_view_0, amount; 0 if missing)
- $u_{i,r}$: unit for $a_{i,r}$ (file_3_view_0, unit)
- $min_c$: minimum total quantity for category $c$ (file_4_view_0, minimum_quantity)
- $max_c$: maximum total quantity for category $c$ (file_4_view_0, maximum_quantity)
- $F_c$: activation fee for category $c$ (file_4_view_0, activation_fee_cents)
- $cap_r$: available capacity for resource $r$ (sum of file_8_view_0, amount, for each $r$)
- $U_r$: unit for $cap_r$ (file_8_view_0, unit)
- $b_{ij}$: bundle bonus in cents for bundle $(i,j)$ (file_7_view_0, bonus_cents)
- $prereq(i)$: set of prerequisite items for $i$ (file_0_view_0, prerequisite_ref)
- $inc(i)$: set of items incompatible with $i$ (file_5_view_0, item_b for item_a $=i$ and item_a for item_b $=i$)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: quantity ordered of item $i \in I$
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise, for $i \in I$
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ with $cat(i)=c$, 0 otherwise, for $c \in C$
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j) \in B$, 0 otherwise

**Objective**
\[
\max \left(
\sum_{i \in I} u_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j) \in B} b_{ij} w_{ij}
\right)
\]

**Subject to**

1. **Item quantity bounds**
   \[
   x_i = 0 \quad \text{or} \quad min_i \leq x_i \leq max_i \qquad \forall i \in I
   \]
   (Enforced via $y_i$ below.)

2. **Item activation logic**
   \[
   x_i \leq max_i y_i \qquad \forall i \in I
   \]
   \[
   x_i \geq min_i y_i \qquad \forall i \in I
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

3. **Category activation logic**
   \[
   z_c \geq y_i \qquad \forall c \in C,\, i \in I: cat(i) = c
   \]
   \[
   z_c \in \{0,1\} \qquad \forall c \in C
   \]

4. **Category quantity bounds**
   \[
   min_c \leq \sum_{i \in I: cat(i)=c} x_i \leq max_c \qquad \forall c \in C
   \]

5. **Resource constraints (unit conversions as needed)**
   - For each resource $r$:
     - Let $S_r$ be the sum of all $a_{i,r} x_i$ for $i \in I$ (convert units as needed: 1 kwh = 1000 wh, 1 liter = 1000 ml, 1 hour = 60 min).
     - Let $cap_r$ be the sum of all file_8_view_0, amount, for $r$ (already signed).
   \[
   \sum_{i \in I} a_{i,r}^* x_i \leq cap_r^* \qquad \forall r \in R
   \]
   where $a_{i,r}^*$ and $cap_r^*$ are all in the same units (convert as per instructions).

6. **Incompatibility constraints**
   \[
   y_i + y_j \leq 1 \qquad \forall (i,j) \in Q
   \]

7. **Prerequisite constraints**
   \[
   y_i \leq y_{p} \qquad \forall (i,p) \in P
   \]

8. **Bundle bonus logic**
   \[
   w_{ij} \leq y_i \qquad \forall (i,j) \in B
   \]
   \[
   w_{ij} \leq y_j \qquad \forall (i,j) \in B
   \]
   \[
   w_{ij} \geq y_i + y_j - 1 \qquad \forall (i,j) \in B
   \]
   \[
   w_{ij} \in \{0,1\} \qquad \forall (i,j) \in B
   \]

9. **Nonnegativity and integrality**
   \[
   x_i \in \mathbb{Z}_+, \quad y_i \in \{0,1\} \qquad \forall i \in I
   \]

**Data Mapping**

- Items $I$, with $u_i$, $f_i$, $cat(i)$, $min_i$, $max_i$: file_2_view_0 (columns item_ref, unit_benefit_cents, item_fee_cents, category, minimum_lot, maximum_order)
- Category constraints $C$, $min_c$, $max_c$, $F_c$: file_4_view_0 (category, minimum_quantity, maximum_quantity, activation_fee_cents)
- Resource usage $a_{i,r}$: file_3_view_0 (item_ref, resource, amount, unit)
- Resource capacities $cap_r$: sum of file_8_view_0 (resource, amount, unit)
- Prerequisites $P$: file_0_view_0 (item_ref, prerequisite_ref)
- Incompatibilities $Q$: file_5_view_0 (item_a, item_b)
- Bundle bonuses $B$, $b_{ij}$: file_7_view_0 (item_a, item_b, bonus_cents)

**Notes**
- All units are converted as per instructions before summing for resource constraints.
- Only authorized=1 items (file_2_view_0) are included in $I$.
- All variables and constraints are indexed over the current data as described above.
- The objective is in USD cents, as all benefit, fee, and bonus values are in cents.

**Maximize the net benefit in USD cents.**