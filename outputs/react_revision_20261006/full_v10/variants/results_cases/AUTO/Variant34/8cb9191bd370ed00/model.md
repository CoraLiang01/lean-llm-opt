Let:
- $I$ = set of item options (item_ref) authorized for display (from export_07.csv and export_08.csv, where authorized=1)
- $R$ = set of resources (display sections), e.g., SECTION_A, SECTION_B, SECTION_C
- $C$ = set of categories (from export_04.csv)
- $B$ = set of bundle pairs (from export_02.csv)
- $Q_i$ = integer number of packs to display for item $i \in I$
- $y_i$ = binary, 1 if $Q_i > 0$, 0 otherwise, for $i \in I$
- $z_c$ = binary, 1 if any $Q_i > 0$ for $i$ in category $c$, 0 otherwise
- $w_{ij}$ = binary, 1 if both $Q_i > 0$ and $Q_j > 0$ for bundle $(i,j) \in B$, 0 otherwise

Parameters (from data mapping below):
- $b_i$ = per-pack net benefit for item $i$ (sum of amount_cents for item_ref $i$ in export_01.csv)
- $f_i$ = item_fee for item $i$ (activation_fee_cents from export_09.csv)
- $cat(i)$ = category of item $i$ (from export_07.csv/export_08.csv)
- $min_i$, $max_i$ = minimum_lot, maximum_order for item $i$ (from export_07.csv/export_08.csv)
- $auth_i$ = 1 if item $i$ is authorized, 0 otherwise (from export_07.csv/export_08.csv)
- $loc(i)$ = location_id (section) for item $i$
- $u_{ir}$ = per-pack resource usage of item $i$ in resource $r$ (sum over export_12.csv and export_13.csv, convert liters to ml)
- $cap_r$ = total available capacity for resource $r$ (sum of amount for resource $r$ in export_03.csv)
- $catmin_c$, $catmax_c$, $catfee_c$ = minimum_quantity, maximum_quantity, activation_fee_cents for category $c$ (from export_04.csv)
- $inc$ = set of incompatible pairs (from export_06.csv)
- $req$ = set of requires pairs (from export_11.csv)
- $bonus_{ij}$ = bundle bonus_cents for $(i,j) \in B$ (from export_02.csv)

Objective:
Maximize net merchandising benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} \left[ b_i Q_i - f_i y_i \right]
+ \sum_{c \in C} \left[ -catfee_c z_c \right]
+ \sum_{(i,j) \in B} bonus_{ij} w_{ij}
\right\}
\]

Subject to:

1. Item selection and bounds:
\[
Q_i = 0 \quad \text{if } auth_i = 0 \qquad \forall i \in I
\]
\[
Q_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z} \qquad \forall i \in I \text{ with } auth_i = 1
\]
\[
y_i = \begin{cases}
1 & Q_i > 0 \\
0 & Q_i = 0
\end{cases} \qquad \forall i \in I
\]

2. Section (resource) capacity:
\[
\sum_{i \in I: loc(i) = r} u_{ir} Q_i \leq cap_r \qquad \forall r \in R
\]

3. Category quantity limits and activation:
\[
catmin_c \leq \sum_{i \in I: cat(i) = c} Q_i \leq catmax_c \qquad \forall c \in C
\]
\[
z_c \geq y_i \qquad \forall i \in I,\, c = cat(i)
\]
\[
z_c \in \{0,1\} \qquad \forall c \in C
\]

4. Incompatibility:
\[
y_i + y_j \leq 1 \qquad \forall (i,j) \in inc
\]

5. Requires:
\[
y_i \leq y_j \qquad \forall (i,j) \in req
\]

6. Bundle bonuses:
\[
w_{ij} \leq y_i,\quad w_{ij} \leq y_j,\quad w_{ij} \geq y_i + y_j - 1 \qquad \forall (i,j) \in B
\]
\[
w_{ij} \in \{0,1\}
\]

7. Variable domains:
\[
Q_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z} \qquad \forall i \in I
\]
\[
y_i \in \{0,1\} \qquad \forall i \in I
\]

Data Mapping:
- file_0_view_0: benefit components per item_ref ($b_i$)
- file_1_view_0: bundle bonuses ($B$, $bonus_{ij}$)
- file_2_view_0: resource capacity ledger ($cap_r$)
- file_3_view_0: category limits and activation fees ($catmin_c$, $catmax_c$, $catfee_c$)
- file_4_view_0: item_ref identity mapping
- file_5_view_0: incompatible item pairs ($inc$)
- file_6_view_0, file_7_view_0: item options for display, bounds, category, location, authorization ($I$, $min_i$, $max_i$, $cat(i)$, $loc(i)$, $auth_i$)
- file_8_view_0: item_ref fixed fee ($f_i$)
- file_10_view_0: requires relationships ($req$)
- file_11_view_0, file_12_view_0: item_ref resource usage ($u_{ir}$, convert liters to ml: $1$ liter $= 1000$ ml)

All sums, bounds, and constraints are over the current authorized item options and categories as defined above. All monetary values are in USD cents.