Mathematical Model for CENTRAL_FRESH Produce Order

Sets:
- $I$: set of authorized items (from file_5_view_0, column item_ref)
- $C$: set of categories (from file_2_view_0, column category)
- $R$: set of resources (from file_1_view_0, column resource)
- $B$: set of bundle pairs (from file_0_view_0, columns item_a, item_b)
- $P$: set of incompatible pairs (from file_4_view_0, columns item_a, item_b)
- $Q$: set of prerequisite pairs (from file_7_view_0, columns item_ref, prerequisite_ref)

Parameters:
- $u_i$: unit_benefit_cents for item $i$ (file_5_view_0)
- $f_i$: item_fee_cents for item $i$ (file_5_view_0)
- $cat(i)$: category of item $i$ (file_5_view_0)
- $min_i$, $max_i$: minimum_lot, maximum_order for item $i$ (file_5_view_0)
- $min_c$, $max_c$: minimum_quantity, maximum_quantity for category $c$ (file_2_view_0)
- $F_c$: activation_fee_cents for category $c$ (file_2_view_0)
- $a_{ir}$: per-unit usage of resource $r$ by item $i$ (file_8_view_0, amount, converted to base units)
- $cap_r$: total available capacity for resource $r$ (sum of file_1_view_0, opening and reservation, converted to base units)
- $bonus_{ij}$: bonus_cents for bundle $(i,j)$ (file_0_view_0)
- $P$: set of incompatible pairs $(i,j)$ (file_4_view_0)
- $Q$: set of prerequisite pairs $(i,k)$ (file_7_view_0, item_ref, prerequisite_ref)

Decision Variables:
- $x_i \in \mathbb{Z}_+$: number of cases of item $i$ to order ($x_i = 0$ or $min_i \leq x_i \leq max_i$)
- $z_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item ordered indicator)
- $y_c \in \{0,1\}$: 1 if any item in category $c$ is ordered, 0 otherwise (category activation indicator)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)$, 0 otherwise (bundle bonus indicator)

Objective:
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} u_i x_i
- \sum_{i \in I} f_i z_i
+ \sum_{(i,j) \in B} bonus_{ij} w_{ij}
- \sum_{c \in C} F_c y_c
\right\}
\]

Subject to:

1. Item order bounds:
\[
x_i = 0 \quad \text{or} \quad min_i \leq x_i \leq max_i \qquad \forall i \in I
\]
\[
z_i = \begin{cases}
1 & \text{if } x_i > 0 \\
0 & \text{if } x_i = 0
\end{cases} \qquad \forall i \in I
\]
(Enforced via: $x_i \geq min_i z_i$, $x_i \leq max_i z_i$)

2. Category quantity bounds:
\[
min_c \leq \sum_{i \in I: cat(i) = c} x_i \leq max_c \qquad \forall c \in C
\]
\[
y_c \geq z_i \qquad \forall i \in I,\, cat(i) = c
\]
($y_c$ is 1 if any $z_i$ in $c$ is 1)

3. Resource capacity constraints (convert all units to base: 1 liter = 1000 ml, 1 hour = 60 min, 1 kwh = 1000 wh):
\[
\sum_{i \in I} a_{ir} x_i \leq cap_r \qquad \forall r \in R
\]

4. Incompatibility constraints:
\[
z_i + z_j \leq 1 \qquad \forall (i,j) \in P
\]

5. Prerequisite constraints:
\[
z_i \leq z_k \qquad \forall (i,k) \in Q
\]

6. Bundle bonus activation:
\[
w_{ij} \leq z_i, \quad w_{ij} \leq z_j, \quad w_{ij} \geq z_i + z_j - 1 \qquad \forall (i,j) \in B
\]

7. Variable domains:
\[
x_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z} \qquad \forall i \in I
\]
\[
z_i, y_c, w_{ij} \in \{0,1\}
\]

Data Mapping:
- file_5_view_0: Items $I$, with columns item_ref ($i$), category, minimum_lot ($min_i$), maximum_order ($max_i$), unit_benefit_cents ($u_i$), item_fee_cents ($f_i$)
- file_2_view_0: Categories $C$, with columns category ($c$), minimum_quantity ($min_c$), maximum_quantity ($max_c$), activation_fee_cents ($F_c$)
- file_8_view_0: Item resource usage $a_{ir}$, with columns item_ref ($i$), resource ($r$), amount (convert to base units), unit
- file_1_view_0: Resource capacity $cap_r$, sum of opening and reservation for each resource, convert to base units
- file_0_view_0: Bundle bonuses $B$, with columns item_a ($i$), item_b ($j$), bonus_cents ($bonus_{ij}$)
- file_4_view_0: Incompatible pairs $P$, columns item_a ($i$), item_b ($j$)
- file_7_view_0: Prerequisite pairs $Q$, columns item_ref ($i$), prerequisite_ref ($k$)

All indices, parameters, and constraints are defined directly from the supplied tables and their columns as described above.