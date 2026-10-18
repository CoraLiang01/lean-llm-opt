Let $x_i$ be the number of units of component $i$ to produce, for $i \in \{C1, C2, \ldots, C111\}$.

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit time required for component $i$ in workshop $w$ (from processing_time_unit.csv, where $w$ is one of: Casting, Milling, Finishing, Assembly, QA & Packaging).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The model is:

Maximize total output value:
$$
\max \sum_{i=1}^{111} p_i x_i
$$

Subject to workshop time constraints:
$$
\sum_{i=1}^{111} a_{wi} x_i \leq b_w, \quad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
$$

Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{C1, C2, \ldots, C111\}
$$

Where the parameters are:

- For each component $i$:

| Component | $p_i$ (unit price) |
|-----------|-------------------|
| C1        | 193               |
| C2        | 64                |
| C3        | 103               |
| ...       | ...               |
| C111      | 142               |

- For each workshop $w$:

| Workshop         | $b_w$ (total_hours) |
|------------------|--------------------|
| Casting          | 7650               |
| Milling          | 6320               |
| Finishing        | 5538               |
| Assembly         | 5957               |
| QA & Packaging   | 6988               |

- For each workshop $w$ and component $i$, $a_{wi}$ is given by the corresponding entry in processing_time_unit.csv. For example:

| Workshop         | C1   | C2   | ... | C111 |
|------------------|------|------|-----|------|
| Casting          | 0.74 | 0.77 | ... | 3.81 |
| Milling          | 0.60 | 3.38 | ... | 2.67 |
| Finishing        | 0.00 | 4.15 | ... | 4.07 |
| Assembly         | 4.84 | 0.00 | ... | 4.02 |
| QA & Packaging   | 0.92 | 0.00 | ... | 3.84 |

All coefficients are as in the retrieved data, preserving original order and identifiers.