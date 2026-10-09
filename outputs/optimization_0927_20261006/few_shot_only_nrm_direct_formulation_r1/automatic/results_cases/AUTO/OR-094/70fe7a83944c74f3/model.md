**Sets and Indices:**
- $w \in \{1,2,3\}$: Workstation index
- $m \in \{1,2,\ldots,101\}$: Radio model index

**Parameters:**
- $t_{w,m}$: Processing time (in minutes) required at workstation $w$ per unit of model $m$ (from CSV)
- $C_w$: Effective daily capacity (in minutes) at workstation $w$:
  - $C_1 = 1440 \times (1 - 0.10) = 1296$
  - $C_2 = 1440 \times (1 - 0.14) = 1238.4$
  - $C_3 = 1440 \times (1 - 0.12) = 1267.2$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: Number of units of model $m$ to produce per day
- $I_w \geq 0$: Idle time at workstation $w$

**Objective:**
\[
\min \sum_{w=1}^3 I_w
\]

**Constraints:**

1. **Idle time definition for each workstation:**
   \[
   I_w = C_w - \sum_{m=1}^{101} t_{w,m} x_m \qquad \forall w \in \{1,2,3\}
   \]
   (Alternatively, $I_w + \sum_{m=1}^{101} t_{w,m} x_m = C_w$)

2. **Nonnegativity of idle time:**
   \[
   I_w \geq 0 \qquad \forall w \in \{1,2,3\}
   \]

3. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m \in \{1,2,\ldots,101\}
   \]

**Numerical Data from CSV (in source order):**

- Workstation 1 (Maintenance_Percent: 10, $C_1 = 1296$)
- Workstation 2 (Maintenance_Percent: 14, $C_2 = 1238.4$)
- Workstation 3 (Maintenance_Percent: 12, $C_3 = 1267.2$)

- For $m=1$ to $101$:
  - $t_{1,m}$: HiFi$m$_Minutes from Workstation 1 row
  - $t_{2,m}$: HiFi$m$_Minutes from Workstation 2 row
  - $t_{3,m}$: HiFi$m$_Minutes from Workstation 3 row

**Complete Model:**

\[
\begin{align*}
\min \quad & I_1 + I_2 + I_3 \\[2ex]
\text{s.t.} \quad
& I_1 + \sum_{m=1}^{101} t_{1,m} x_m = 1296 \\
& I_2 + \sum_{m=1}^{101} t_{2,m} x_m = 1238.4 \\
& I_3 + \sum_{m=1}^{101} t_{3,m} x_m = 1267.2 \\
& I_w \geq 0 \qquad \forall w \in \{1,2,3\} \\
& x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m \in \{1,2,\ldots,101\}
\end{align*}
\]

Where:
- $t_{w,m}$ are the coefficients from the CSV, as described above.
- $I_w$ are continuous nonnegative variables.
- $x_m$ are nonnegative integers.