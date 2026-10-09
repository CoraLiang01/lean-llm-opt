Let $x_i$ be the number of batches of product $i$ to produce or purchase, for $i = 1, \ldots, 100$. Each batch is exactly 10 units. All variables $x_i$ are nonnegative integers.

Let the following parameters be defined for each product $i$ (using the identifiers and coefficients from the data):

- $p_i$: profit per unit of product $i$
- $a_{i,1}$: units of resource R1 consumed per unit of product $i$
- $a_{i,2}$: units of resource R2 consumed per unit of product $i$
- $a_{i,3}$: units of resource R3 consumed per unit of product $i$
- $d_i$: upper demand bound (units) for product $i$
- $b_i$: batch size in units for product $i$ (here, always 10)

Let $C_1$, $C_2$, $C_3$ be the total available amounts of resources R1, R2, R3, respectively.

The complete model is:

---

**Decision variables:**

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,100\}
$$

---

**Objective:**

$$
\max \sum_{i=1}^{100} 10 \cdot x_i \cdot p_i
$$

---

**Resource constraints:**

$$
\sum_{i=1}^{100} 10 \cdot a_{i,1} \cdot x_i \leq C_1
$$

$$
\sum_{i=1}^{100} 10 \cdot a_{i,2} \cdot x_i \leq C_2
$$

$$
\sum_{i=1}^{100} 10 \cdot a_{i,3} \cdot x_i \leq C_3
$$

---

**Demand upper bound constraints:**

$$
10 \cdot x_i \leq d_i \qquad \forall i \in \{1,2,\ldots,100\}
$$

---

**Parameter values (source order):**

| product | $p_i$ | $a_{i,1}$ | $a_{i,2}$ | $a_{i,3}$ | $d_i$ | $b_i$ |
|---------|-------|-----------|-----------|-----------|-------|-------|
| P1   | 6.7   | 2.37  | 0.61  | 2.03  | 317 | 10 |
| P2   | 10.96 | 4.79  | 2.73  | 0.53  | 106 | 10 |
| P3   | 9.4   | 3.87  | 1.6   | 0.74  | 386 | 10 |
| P4   | 11.13 | 3.31  | 2.28  | 2.73  | 441 | 10 |
| P5   | 10.29 | 1.46  | 3.68  | 1.94  | 63  | 10 |
| P6   | 8.06  | 1.46  | 1.37  | 0.32  | 221 | 10 |
| P7   | 6.94  | 1.04  | 1.94  | 0.57  | 441 | 10 |
| P8   | 11.44 | 4.44  | 3.14  | 2.09  | 489 | 10 |
| P9   | 9.13  | 3.32  | 1.3   | 0.31  | 277 | 10 |
| P10  | 7.83  | 3.77  | 0.77  | 0.73  | 121 | 10 |
| ...  | ...   | ...   | ...   | ...   | ... | ... |
| P100 | 8.84  | 1.25  | 3.23  | 0.53  | 576 | 10 |

(Resource and product lists continue in the same order as above.)

**Resource capacities:**

- $C_1 = 27380.54$
- $C_2 = 22245.11$
- $C_3 = 15147.73$

---

**Variable domains:**

$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,100\}
$$

---

**Summary of constraints:**

- Each $x_i$ is the number of batches (of 10 units) of product $i$ to produce/purchase.
- Total resource consumption across all products cannot exceed the available capacity for each resource.
- For each product, total produced units ($10 x_i$) cannot exceed the upper demand bound $d_i$.
- All variables are nonnegative integers.

All coefficients and identifiers are as retrieved and appear in the same order as the source data.