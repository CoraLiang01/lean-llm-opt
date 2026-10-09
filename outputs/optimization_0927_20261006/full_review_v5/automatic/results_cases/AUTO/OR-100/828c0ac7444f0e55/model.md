Let $x_i$ denote the number of units to produce of component $i$ ($i \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$). All $x_i$ are nonnegative integers.

Let $p_i$ be the unit price of component $i$.

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ ($w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$).

Let $b_w$ be the total available working hours in workshop $w$.

---

#### Objective

$$
\max \sum_{i \in \{\text{C1},\ldots,\text{C111}\}} p_i x_i
$$

#### Constraints

For each workshop $w$:
$$
\sum_{i \in \{\text{C1},\ldots,\text{C111}\}} a_{wi} x_i \leq b_w
$$

For all $i$:
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

---

#### Parameter Tables

**Unit Processing Time $a_{wi}$ (hours per unit):**

| Workshop           | C1   | C2   | C3   | C4   | C5   | ... | C111 |
|--------------------|------|------|------|------|------|-----|------|
| Casting            | 0.74 | 0.77 | 1.41 | 2.11 | 2.19 | ... | 3.81 |
| Milling            | 0.60 | 3.38 | 0.00 | 0.00 | 1.25 | ... | 2.67 |
| Finishing          | 0.00 | 4.15 | 0.00 | 2.45 | 2.49 | ... | 4.07 |
| Assembly           | 4.84 | 0.00 | 3.80 | 2.01 | 3.03 | ... | 4.02 |
| QA & Packaging     | 0.92 | 0.00 | 1.08 | 3.39 | 2.54 | ... | 3.21 |

**Unit Price $p_i$:**

| Component | Unit Price |
|-----------|------------|
| C1        | 193        |
| C2        | 64         |
| C3        | 103        |
| C4        | 210        |
| C5        | 85         |
| ...       | ...        |
| C111      | 142        |

**Total Working Hours $b_w$:**

| Workshop         | Total Hours |
|------------------|------------|
| Casting          | 7650       |
| Milling          | 6320       |
| Finishing        | 5538       |
| Assembly         | 5957       |
| QA & Packaging   | 6988       |

---

#### Complete Model (Numerical Form)

Let $x_i$ for $i \in \{\text{C1}, \ldots, \text{C111}\}$.

**Objective:**
$$
\max \left(
193x_{\text{C1}} + 64x_{\text{C2}} + 103x_{\text{C3}} + 210x_{\text{C4}} + 85x_{\text{C5}} + \cdots + 142x_{\text{C111}}
\right)
$$

**Subject to:**

_Casting:_
$$
0.74x_{\text{C1}} + 0.77x_{\text{C2}} + 1.41x_{\text{C3}} + \cdots + 3.81x_{\text{C111}} \leq 7650
$$

_Milling:_
$$
0.60x_{\text{C1}} + 3.38x_{\text{C2}} + 0.00x_{\text{C3}} + \cdots + 2.67x_{\text{C111}} \leq 6320
$$

_Finishing:_
$$
0.00x_{\text{C1}} + 4.15x_{\text{C2}} + 0.00x_{\text{C3}} + \cdots + 4.07x_{\text{C111}} \leq 5538
$$

_Assembly:_
$$
4.84x_{\text{C1}} + 0.00x_{\text{C2}} + 3.80x_{\text{C3}} + \cdots + 4.02x_{\text{C111}} \leq 5957
$$

_QA & Packaging:_
$$
0.92x_{\text{C1}} + 0.00x_{\text{C2}} + 1.08x_{\text{C3}} + \cdots + 3.21x_{\text{C111}} \leq 6988
$$

_Nonnegativity and integrality:_
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{\text{C1}, \ldots, \text{C111}\}
$$

---

All coefficients and identifiers are as retrieved and in source order.