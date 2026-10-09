Let $x_i$ denote the number of units to produce of component $i$ ($i = \text{C1}, \ldots, \text{C111}$).

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time (in hours) required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in$ {Casting, Milling, Finishing, Assembly, QA & Packaging}).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

**Objective:**
\[
\max \sum_{i=\text{C1}}^{\text{C111}} p_i x_i
\]

**Subject to:**

For each workshop $w$:
\[
\sum_{i=\text{C1}}^{\text{C111}} a_{wi} x_i \leq b_w
\]

\[
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i = \text{C1},\ldots,\text{C111}
\]

---

### Numerical Formulation

**Variables:**
- $x_i$: number of units to produce of component $i$ ($i = \text{C1}, \ldots, \text{C111}$), integer, $x_i \geq 0$

**Parameters:**

From **unit_price.csv**:

| $i$   | $p_i$ |
|-------|-------|
| C1    | 193   |
| C2    | 64    |
| C3    | 103   |
| C4    | 210   |
| C5    | 85    |
| C6    | 126   |
| C7    | 226   |
| C8    | 94    |
| C9    | 73    |
| C10   | 120   |
| ...   | ...   |
| C111  | 142   |

From **processing_time_unit.csv** (partial illustration):

| Workshop           | C1   | C2   | ... | C111 |
|--------------------|------|------|-----|------|
| Casting            | 0.74 | 0.77 | ... | 3.81 |
| Milling            | 0.60 | 3.38 | ... | 2.67 |
| Finishing          | 0.00 | 4.15 | ... | 4.07 |
| Assembly           | 4.84 | 0.00 | ... | 4.02 |
| QA & Packaging     | 0.92 | 0.00 | ... | 3.21 |

From **total_working_hours.csv**:

| Workshop         | $b_w$ |
|------------------|-------|
| Casting          | 7650  |
| Milling          | 6320  |
| Finishing        | 5538  |
| Assembly         | 5957  |
| QA & Packaging   | 6988  |

**Full Model:**

\[
\max \left(
193x_{\text{C1}} + 64x_{\text{C2}} + 103x_{\text{C3}} + 210x_{\text{C4}} + 85x_{\text{C5}} + 126x_{\text{C6}} + 226x_{\text{C7}} + 94x_{\text{C8}} + 73x_{\text{C9}} + 120x_{\text{C10}} + \cdots + 142x_{\text{C111}}
\right)
\]

Subject to:

(Casting constraint)
\[
0.74x_{\text{C1}} + 0.77x_{\text{C2}} + 1.41x_{\text{C3}} + \cdots + 3.81x_{\text{C111}} \leq 7650
\]

(Milling constraint)
\[
0.60x_{\text{C1}} + 3.38x_{\text{C2}} + 0.00x_{\text{C3}} + \cdots + 2.67x_{\text{C111}} \leq 6320
\]

(Finishing constraint)
\[
0.00x_{\text{C1}} + 4.15x_{\text{C2}} + 0.00x_{\text{C3}} + \cdots + 4.07x_{\text{C111}} \leq 5538
\]

(Assembly constraint)
\[
4.84x_{\text{C1}} + 0.00x_{\text{C2}} + 3.80x_{\text{C3}} + \cdots + 4.02x_{\text{C111}} \leq 5957
\]

(QA & Packaging constraint)
\[
0.92x_{\text{C1}} + 0.00x_{\text{C2}} + 1.08x_{\text{C3}} + \cdots + 3.21x_{\text{C111}} \leq 6988
\]

\[
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i = \text{C1},\ldots,\text{C111}
\]

---

**All coefficients and bounds are as given in the source CSVs.**