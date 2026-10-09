## Symbolic Mathematical Model

Let:
- $I$ = set of valid items (indexed by $i$), each with unique item_ref.
- $C$ = set of categories (indexed by $c$).
- $R$ = set of resources (indexed by $r$).
- $B$ = set of valid bundle pairs $(i,j)$.
- $Q_i$ = integer quantity ordered of item $i$ (decision variable).
- $z_c$ = binary, 1 if any item in category $c$ is ordered, 0 otherwise.
- $y_i$ = binary, 1 if $Q_i > 0$, 0 otherwise.

### Parameters (all mapped to source tables/columns):

- $a_i$ = 1 if item $i$ is authorized, 0 otherwise (file_14_view_0, file_15_view_0, column 'authorized').
- $l_i$, $u_i$ = minimum_lot, maximum_order for item $i$ (file_14_view_0, file_15_view_0).
- $c(i)$ = category of item $i$ (file_14_view_0, file_15_view_0).
- $f_i$ = item_fee for item $i$ (file_16_view_0, file_17_view_0).
- $F_c$ = activation_fee_cents for category $c$ (file_7_view_0, file_8_view_0).
- $L_c$, $U_c$ = minimum_quantity, maximum_quantity for category $c$ (file_7_view_0, file_8_view_0).
- $b_{ij}$ = bundle bonus_cents for bundle $(i,j)$ (file_3_view_0, file_4_view_0).
- $S$ = set of incompatible pairs $(i,j)$ (file_12_view_0, file_13_view_0).
- $D$ = set of requires pairs $(i,k)$ (file_20_view_0, file_21_view_0).
- $u_{ir}$ = usage of resource $r$ per unit of item $i$ (file_22_view_0, file_23_view_0, file_24_view_0).
- $K_r$ = total available capacity for resource $r$ (sum of signed amounts in file_5_view_0, file_6_view_0 for $r$).
- $s_{ij}$ = 1 if $(i,j)$ is an incompatible pair, 0 otherwise.
- $d_{ik}$ = 1 if item $i$ requires item $k$, 0 otherwise.
- $v_{ic}$ = 1 if $c(i)=c$, 0 otherwise.
- $B_i$ = per-unit net benefit in USD cents for item $i$ (sum of all benefit components for $i$ after currency conversion using file_9_view_0).

### Unit conversions:
- For resource usage and capacity: 1 liter = 1000 ml, 1 hour = 60 minutes, 1 kwh = 1000 wh.

### Decision Variables:
- $Q_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.
- $y_i \in \{0,1\}$, for all $i \in I$.
- $z_c \in \{0,1\}$, for all $c \in C$.

### Objective:
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} B_i Q_i
- \sum_{i \in I} f_i y_i
+ \sum_{(i,j) \in B} b_{ij} \cdot \min\{y_i, y_j\}
- \sum_{c \in C} F_c z_c
\right\}
\]

### Constraints:

1. **Authorization and Lot Constraints:**
   \[
   l_i y_i \leq Q_i \leq u_i y_i \quad \forall i \in I
   \]
   \[
   Q_i = 0 \quad \text{if } a_i = 0
   \]
   \[
   y_i = 1 \iff Q_i > 0
   \]
   \[
   y_i \in \{0,1\}
   \]

2. **Category Quantity Limits:**
   \[
   L_c z_c \leq \sum_{i \in I: c(i)=c} Q_i \leq U_c z_c \quad \forall c \in C
   \]
   \[
   z_c \geq y_i \quad \forall i \in I, c = c(i)
   \]

3. **Resource Capacity:**
   \[
   \sum_{i \in I} u_{ir} Q_i \leq K_r \quad \forall r \in R
   \]
   (Convert all $u_{ir}$ and $K_r$ to common units.)

4. **Incompatibility:**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in S
   \]

5. **Requires Dependencies:**
   \[
   Q_i \leq M \cdot y_k \quad \forall (i,k) \in D
   \]
   (where $M$ is a large constant, or more strictly $Q_k \geq 1$ if $Q_i \geq 1$.)

6. **Bundle Bonuses:**
   \[
   \text{Award } b_{ij} \text{ once if } y_i = y_j = 1, \text{ zero otherwise.}
   \]

7. **Variable Domains:**
   \[
   Q_i \in \mathbb{Z}_{\geq 0},\quad y_i \in \{0,1\},\quad z_c \in \{0,1\}
   \]

### Data Mapping

- All sets, parameters, and variables are mapped to the latest valid records as of 2026-05-07, per the selection rules:
  - For each (dealership_id, table, record_id), keep the highest integer revision with effective_date $\leq$ 2026-05-07, discard if DELETE.
  - Use only records with dealership_id = 'OSLO_NEW_CARS'.
  - For benefit components: file_0_view_0, file_1_view_0, file_2_view_0.
  - For bundle bonuses: file_3_view_0, file_4_view_0.
  - For incompatible pairs: file_12_view_0, file_13_view_0.
  - For requires dependencies: file_20_view_0, file_21_view_0.
  - For item master: file_14_view_0, file_15_view_0.
  - For category limits: file_7_view_0, file_8_view_0.
  - For item fees: file_16_view_0, file_17_view_0.
  - For usage per item: file_22_view_0, file_23_view_0, file_24_view_0.
  - For capacity ledger: file_5_view_0, file_6_view_0.
  - For fx rates: file_9_view_0.
- All currency conversions use amount $\times$ usd_cents_numerator / denominator, with fx rates from file_9_view_0.
- All resource units are converted as specified (liter/ml, hour/minute, kwh/wh).

### Maximum Net Benefit

The maximum net benefit in USD cents is the optimal value of the above objective, subject to all constraints.

---

**This model, with all sets and parameters mapped as above, yields the vehicle order with the largest net benefit for the Oslo dealership as of 2026-05-07.**