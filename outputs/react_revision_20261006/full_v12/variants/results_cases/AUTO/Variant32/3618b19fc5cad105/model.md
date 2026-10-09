## Symbolic Mathematical Model

Let:
- $I$ = set of valid items (options) as of 2026-05-07, after applying the selection rule to all tables.
- $C$ = set of categories.
- $R$ = set of resources.
- $B$ = set of valid bundle pairs $(i,j)$.
- $Q_i$ = integer quantity ordered of item $i \in I$.
- $y_c$ = binary, 1 if any item in category $c$ is ordered, 0 otherwise.
- $z_i$ = binary, 1 if $Q_i > 0$, 0 otherwise.

Parameters (all as of 2026-05-07, after selection rule):
- $a_i$ = 1 if item $i$ is authorized, 0 otherwise.
- $l_i$, $u_i$ = minimum_lot, maximum_order for item $i$.
- $cat(i)$ = category of item $i$.
- $f_{cat}$ = activation_fee_cents for category $cat$.
- $f_i$ = item_fee for item $i$.
- $b_{ij}$ = bundle bonus_cents for bundle $(i,j)$.
- $s_{ij}$ = 1 if $(i,j)$ is an incompatible pair, 0 otherwise.
- $d_{ij}$ = 1 if $i$ requires $j$, 0 otherwise.
- $L_c$, $U_c$ = minimum_quantity, maximum_quantity for category $c$.
- $r_{ir}$ = resource usage per unit of item $i$ for resource $r$ (in native units).
- $K_r$ = total available resource $r$ (in native units).
- $fx_{cur}$ = USD cents per unit of currency $cur$.
- $v_{ik}$ = benefit component $k$ for item $i$ (in native units, converted to USD cents).
- $F_i$ = sum of all $v_{ik}$ for item $i$ (total per-unit benefit in USD cents).

Objective:
$$
\max \left\{ \sum_{i \in I} Q_i F_i - \sum_{i \in I} f_i z_i + \sum_{c \in C} f_{cat} y_c + \sum_{(i,j) \in B} b_{ij} \cdot \min\{z_i, z_j\} \right\}
$$

Subject to:

**Authorization and lot constraints:**
$$
Q_i = 0 \quad \text{if } a_i = 0 \qquad \forall i \in I
$$
$$
Q_i = 0 \text{ or } l_i \leq Q_i \leq u_i \qquad \forall i \in I \text{ with } a_i = 1
$$
$$
Q_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

**Category quantity limits:**
$$
L_c \leq \sum_{i:cat(i)=c} Q_i \leq U_c \qquad \forall c \in C
$$

**Category activation:**
$$
y_c \geq z_i \qquad \forall i \in I,\, c = cat(i)
$$
$$
y_c \in \{0,1\} \qquad \forall c \in C
$$

**Item activation:**
$$
z_i = \begin{cases}
1 & \text{if } Q_i > 0 \\
0 & \text{if } Q_i = 0
\end{cases} \qquad \forall i \in I
$$
$$
z_i \in \{0,1\} \qquad \forall i \in I
$$

**Resource constraints (with unit conversions):**
Let $r_{ir}$ and $K_r$ be converted to a common base unit:
- For $space$: 1 liter = 1000 ml
- For $labor$: 1 hour = 60 minutes
- For $power$: 1 kwh = 1000 wh

$$
\sum_{i \in I} Q_i \cdot r_{ir}^* \leq K_r^* \qquad \forall r \in R
$$
where $r_{ir}^*$ and $K_r^*$ are in the base units.

**Incompatibility:**
$$
z_i + z_j \leq 1 \qquad \forall (i,j): s_{ij} = 1
$$

**Requires dependencies:**
$$
Q_i \leq u_i \cdot z_j \qquad \forall (i,j): d_{ij} = 1
$$

**Bundle bonuses:**
Award $b_{ij}$ once if both $z_i = 1$ and $z_j = 1$, zero otherwise.

**Summary of Data Mapping:**
- All sets, parameters, and variables are defined from the valid records as of 2026-05-07, after applying the selection rule (exclude future, keep max revision, drop DELETE).
- All benefit components are summed per item, converted to USD cents using $fx$ (amount $\times$ usd_cents_numerator / denominator).
- All resource usage and capacity are converted to base units.
- All category and item fees, bonuses, and limits are as per the latest valid records.
- All constraints and summations are over the current valid entities.

**Report:**
Return the maximum net benefit in USD cents.