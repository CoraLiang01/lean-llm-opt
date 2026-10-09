Let:
- $I$ = set of valid items (indexed by $i$), each with unique item_ref.
- $C$ = set of categories (indexed by $c$).
- $R$ = set of resources (indexed by $r$).
- $B$ = set of valid bundle pairs $(i,j)$.
- $Q_i$ = integer quantity ordered of item $i$ (decision variable).
- $Y_c$ = binary, 1 if any item in category $c$ is ordered, 0 otherwise.
- $Z_i$ = binary, 1 if $Q_i > 0$, 0 otherwise.

Parameters (all as of 2026-05-07, after applying the selection rules):
- $a_i$ = 1 if item $i$ is authorized, 0 otherwise.
- $l_i$, $u_i$ = minimum_lot, maximum_order for item $i$.
- $cat(i)$ = category of item $i$.
- $L_c$, $U_c$ = minimum_quantity, maximum_quantity for category $c$.
- $F_c$ = activation_fee_cents for category $c$.
- $f_i$ = item_fee for item $i$ (activation_fee_cents).
- $b_{ij}$ = bundle bonus_cents for bundle $(i,j)$.
- $inc(i,j)$ = 1 if $(i,j)$ is an incompatible pair, 0 otherwise.
- $req(i,k)$ = 1 if $i$ requires $k$, 0 otherwise.
- $v_{i,s}$ = benefit component $s$ for item $i$ (after currency conversion).
- $S_i$ = set of benefit components for item $i$.
- $u_{i,r}$ = usage of resource $r$ per unit of item $i$ (in resource units).
- $cap_r$ = available capacity for resource $r$ (in resource units).
- $fx(c)$ = USD cents per unit of currency $c$ (from fx table).

Objective:
Maximize net benefit in USD cents:
\[
\max \sum_{i \in I} \left( \sum_{s \in S_i} v_{i,s} \right) Q_i
- \sum_{i \in I} f_i Z_i
- \sum_{c \in C} F_c Y_c
+ \sum_{(i,j) \in B} b_{ij} \cdot \min\{Z_i, Z_j\}
\]

Subject to:

1. Item authorization and lot size:
\[
Q_i = 0 \quad \text{if } a_i = 0
\]
\[
Q_i = 0 \text{ or } l_i \leq Q_i \leq u_i \quad \text{if } a_i = 1
\]
\[
Q_i \in \mathbb{Z}_{\geq 0}
\]
\[
Z_i = 1 \iff Q_i > 0
\]

2. Category quantity limits (unconditional):
\[
L_c \leq \sum_{i: cat(i) = c} Q_i \leq U_c \quad \forall c \in C
\]
\[
Y_c = 1 \iff \sum_{i: cat(i) = c} Q_i > 0
\]

3. Resource capacity (per resource, after unit conversion):
\[
\sum_{i \in I} u_{i,r} Q_i \leq cap_r \quad \forall r \in R
\]

4. Incompatibility:
\[
Q_i = 0 \text{ or } Q_j = 0 \quad \forall (i,j) \text{ with } inc(i,j) = 1
\]

5. Requires dependencies:
\[
Q_i > 0 \implies Q_k > 0 \quad \forall (i,k) \text{ with } req(i,k) = 1
\]

6. Bundle bonuses:
\[
\text{Award } b_{ij} \text{ once if } Q_i > 0 \text{ and } Q_j > 0, \text{ else 0}
\]

7. Each item_fee $f_i$ is deducted once if $Q_i > 0$.

8. Each category activation fee $F_c$ is deducted once if any $Q_i > 0$ for $i$ in $c$.

9. All benefit components for each item are summed after currency conversion:
\[
v_{i,s} = \text{amount}_{i,s} \cdot \frac{\text{usd\_cents\_numerator}}{\text{denominator}} \cdot fx(\text{currency}_{i,s})
\]

10. Resource usage per item is converted to the resource's base unit:
- $1$ liter = $1000$ ml
- $1$ hour = $60$ minutes
- $1$ kwh = $1000$ wh

Data Mapping:
- Items, categories, resources, bundles, incompatibilities, requires, benefit components, item fees, category limits, resource usage, and fx rates are mapped from the respective tables (see table_id and column names in the Observation).
- All selection rules (latest revision, no DELETE, no future effective_date) are applied per (dealership_id, table, record_id) before joining.

Decision variables:
- $Q_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$
- $Z_i \in \{0,1\}$ for all $i \in I$
- $Y_c \in \{0,1\}$ for all $c \in C$

Report:
- The maximum net benefit in USD cents.