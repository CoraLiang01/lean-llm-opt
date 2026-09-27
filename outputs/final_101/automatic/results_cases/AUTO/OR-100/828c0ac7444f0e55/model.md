Let $x_i$ be the number of units to produce of component $i$ ($i \in \{C1, C2, \ldots, C111\}$), where $x_i \in \mathbb{Z}_{\geq 0}$.

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$).

Let $b_w$ be the total available working hours for workshop $w$ (from total_working_hours.csv).

The complete model is:

Objective:
$$
\max \sum_{i=1}^{111} p_i x_i
$$

Subject to (for each workshop $w$):

$$
\sum_{i=1}^{111} a_{wi} x_i \leq b_w \qquad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{C1, C2, \ldots, C111\}
$$

Where the coefficients are as follows (all values as retrieved, in source order):

#### Unit processing times $a_{wi}$ (hours per unit):

| Workshop           | C1   | C2   | C3   | ... | C111 |
|--------------------|------|------|------|-----|------|
| Casting            | 0.74 | 0.77 | 1.41 | ... | 3.81 |
| Milling            | 0.60 | 3.38 | 0.00 | ... | 2.67 |
| Finishing          | 0.00 | 4.15 | 0.00 | ... | 4.07 |
| Assembly           | 4.84 | 0.00 | 3.80 | ... | 4.02 |
| QA & Packaging     | 0.92 | 0.00 | 1.08 | ... | 3.21 |

#### Unit prices $p_i$:

| Component | Unit Price |
|-----------|------------|
| C1        | 193        |
| C2        | 64         |
| C3        | 103        |
| ...       | ...        |
| C111      | 142        |

#### Total available working hours $b_w$:

| Workshop           | Total Hours |
|--------------------|-------------|
| Casting            | 7650        |
| Milling            | 6320        |
| Finishing          | 5538        |
| Assembly           | 5957        |
| QA & Packaging     | 6988        |

All coefficients and identifiers are as above, in the order retrieved. Each $x_i$ is a nonnegative integer. The model maximizes total output value subject to the available hours in each workshop.