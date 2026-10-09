Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of storage areas, indexed by StorageID:
  $$
  S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}
  $$
  with capacities:
  \begin{align*}
  \text{Capacity}_1 &= 1083 \\
  \text{Capacity}_2 &= 1840 \\
  \text{Capacity}_3 &= 770 \\
  \text{Capacity}_4 &= 1299 \\
  \text{Capacity}_5 &= 1259 \\
  \text{Capacity}_6 &= 543 \\
  \text{Capacity}_7 &= 1831 \\
  \text{Capacity}_8 &= 855 \\
  \text{Capacity}_9 &= 619 \\
  \text{Capacity}_{10} &= 637 \\
  \text{Capacity}_{11} &= 935 \\
  \text{Capacity}_{12} &= 626 \\
  \text{Capacity}_{13} &= 1457 \\
  \text{Capacity}_{14} &= 1198 \\
  \text{Capacity}_{15} &= 837 \\
  \end{align*}

- Let $P$ be the set of air conditioner types, indexed by ProductName:
  $$
  P = \{\text{Window Unit},\ \text{Portable Unit},\ \text{Split System},\ \text{Ductless System},\ \text{Central AC},\ \text{Hybrid AC},\ \text{Geothermal AC},\ \text{Smart AC},\ \text{Evaporative Cooler},\ \text{Package Unit}\}
  $$
  with values and weights:
  \begin{align*}
  \text{Window Unit:} &\quad \text{Value} = 4811,\ \text{Weight} = 114 \\
  \text{Portable Unit:} &\quad \text{Value} = 1130,\ \text{Weight} = 200 \\
  \text{Split System:} &\quad \text{Value} = 1611,\ \text{Weight} = 106 \\
  \text{Ductless System:} &\quad \text{Value} = 3368,\ \text{Weight} = 256 \\
  \text{Central AC:} &\quad \text{Value} = 2135,\ \text{Weight} = 268 \\
  \text{Hybrid AC:} &\quad \text{Value} = 1046,\ \text{Weight} = 185 \\
  \text{Geothermal AC:} &\quad \text{Value} = 4030,\ \text{Weight} = 299 \\
  \text{Smart AC:} &\quad \text{Value} = 3761,\ \text{Weight} = 131 \\
  \text{Evaporative Cooler:} &\quad \text{Value} = 3523,\ \text{Weight} = 139 \\
  \text{Package Unit:} &\quad \text{Value} = 1701,\ \text{Weight} = 105 \\
  \end{align*}

---

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
$$

**Objective:**
$$
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i \in S$:
$$
\sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Where:**

- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$
- $\text{Value}_j$ = value of air conditioner type $j$ (see above)
- $\text{Weight}_j$ = size (weight) of air conditioner type $j$ (see above)
- $\text{Capacity}_i$ = capacity of storage area $i$ (see above)

All identifiers and coefficients are as retrieved and preserved in source order.