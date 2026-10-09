**Sets and Indices:**
- $w \in \{1,2,3\}$: Workstation index
- $m \in \{1,2,\ldots,101\}$: Radio model index (corresponding to HiFi1, HiFi2, ..., HiFi101)

**Parameters:**
- $T = 1440$ (total minutes per workstation per day)
- Maintenance percentages:
  - $p_1 = 0.10$
  - $p_2 = 0.14$
  - $p_3 = 0.12$
- Effective capacities:
  - $C_1 = 1440 \times (1 - 0.10) = 1296$
  - $C_2 = 1440 \times (1 - 0.14) = 1238.4$
  - $C_3 = 1440 \times (1 - 0.12) = 1267.2$
- $a_{wm}$: Processing time (in minutes) required for one unit of model $m$ at workstation $w$ (from CSV columns HiFi$m$_Minutes for each workstation $w$).

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: Number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: Idle time (in minutes) at workstation $w$

---

**Mathematical Model:**

Minimize total idle time:
$$
\min \sum_{w=1}^3 I_w
$$

Subject to, for each workstation $w$:
$$
\sum_{m=1}^{101} a_{wm} x_m + I_w = C_w \qquad \forall w \in \{1,2,3\}
$$

$$
x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m \in \{1,2,\ldots,101\}
$$

$$
I_w \geq 0 \qquad \forall w \in \{1,2,3\}
$$

---

**Parameter Values from CSV (in source order):**

- Workstation 1 (Maintenance_Percent: 10, $C_1 = 1296$):
  - HiFi1_Minutes: 6
  - HiFi2_Minutes: 4
  - ...
  - HiFi101_Minutes: 9

- Workstation 2 (Maintenance_Percent: 14, $C_2 = 1238.4$):
  - HiFi1_Minutes: 5
  - HiFi2_Minutes: 5
  - ...
  - HiFi101_Minutes: 3

- Workstation 3 (Maintenance_Percent: 12, $C_3 = 1267.2$):
  - HiFi1_Minutes: 4
  - HiFi2_Minutes: 6
  - ...
  - HiFi101_Minutes: 6

(Use the exact values from the CSV for $a_{wm}$ for all $w$ and $m$.)

---

**Complete Model:**

Minimize
$$
I_1 + I_2 + I_3
$$

Subject to:
\[
\begin{align*}
6x_1 + 4x_2 + 6x_3 + \cdots + 9x_{101} + I_1 &= 1296 \\
5x_1 + 5x_2 + 5x_3 + \cdots + 3x_{101} + I_2 &= 1238.4 \\
4x_1 + 6x_2 + 5x_3 + \cdots + 6x_{101} + I_3 &= 1267.2 \\
x_m &\in \mathbb{Z}_{\geq 0} \quad \forall m=1,\ldots,101 \\
I_w &\geq 0 \quad \forall w=1,2,3
\end{align*}
\]

Where $x_m$ is the number of units of HiFi$m$ to produce per day, and $a_{wm}$ are the processing times from the CSV for each workstation and model.