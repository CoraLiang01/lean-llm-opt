##### Sets and Indices

- Products: $P = \{\text{I}, \text{II}, \text{III}\}$
- Procedure A Equipment: $A = \{\text{A1}, \text{A2}\}$
- Procedure B Equipment: $B = \{\text{B1}, \text{B2}, \text{B3}\}$
- Let $x_{p}$: total production quantity of product $p$
- Let $x_{p,e}$: quantity of product $p$ processed on equipment $e$ (for eligible $p,e$ pairs)

##### Parameters (from CSV)

- Processing time per unit (minutes/unit):

  | Equipment | Product I | Product II | Product III |
  |-----------|-----------|------------|-------------|
  | A1        | 5         | 10         | —           |
  | A2        | 7         | 9          | 12          |
  | B1        | 6         | 8          | —           |
  | B2        | 4         | —          | 11          |
  | B3        | 7         | —          | —           |

- Available Equipment Operating Time (minutes):

  | Equipment | Available Time |
  |-----------|---------------|
  | A1        | 6000          |
  | A2        | 10000         |
  | B1        | 4000          |
  | B2        | 7000          |
  | B3        | 4000          |

- Equipment Cost at Full Load (yuan):

  | Equipment | Cost |
  |-----------|------|
  | A1        | 300  |
  | A2        | 321  |
  | B1        | 250  |
  | B2        | 783  |
  | B3        | 200  |

- Raw Material Cost (yuan/unit):

  | Product   | Cost |
  |-----------|------|
  | I         | 0.25 |
  | II        | 0.35 |
  | III       | 0.5  |

- Unit Price (yuan/unit):

  | Product   | Price |
  |-----------|-------|
  | I         | 1.25  |
  | II        | 2     |
  | III       | 2.8   |

##### Eligibility Constraints

- Product I: A1 or A2 for A; B1, B2, or B3 for B
- Product II: A1 or A2 for A; only B1 for B
- Product III: only A2 for A; only B2 for B

##### Decision Variables

- $x_{p,e} \geq 0$: quantity of product $p$ processed on equipment $e$ (only for eligible pairs)

##### Objective Function

Maximize total profit:

\[
\max \left\{
\sum_{p \in P} \left[ (\text{Unit Price}_p - \text{Raw Material Cost}_p) \cdot x_p \right]
- \sum_{e \in A \cup B} \text{Equipment Cost}_e
\right\}
\]

where $x_p$ is the total production of product $p$ (see below).

##### Constraints

###### 1. Production Consistency

For each product, total production equals sum over eligible equipment for each procedure:

- For Product I:
  - $x_{\text{I}} = x_{\text{I,A1}} + x_{\text{I,A2}} = x_{\text{I,B1}} + x_{\text{I,B2}} + x_{\text{I,B3}}$
- For Product II:
  - $x_{\text{II}} = x_{\text{II,A1}} + x_{\text{II,A2}} = x_{\text{II,B1}}$
- For Product III:
  - $x_{\text{III}} = x_{\text{III,A2}} = x_{\text{III,B2}}$

###### 2. Equipment Time Constraints

For each equipment, total processing time cannot exceed available time:

- A1: $5 x_{\text{I,A1}} + 10 x_{\text{II,A1}} \leq 6000$
- A2: $7 x_{\text{I,A2}} + 9 x_{\text{II,A2}} + 12 x_{\text{III,A2}} \leq 10000$
- B1: $6 x_{\text{I,B1}} + 8 x_{\text{II,B1}} \leq 4000$
- B2: $4 x_{\text{I,B2}} + 11 x_{\text{III,B2}} \leq 7000$
- B3: $7 x_{\text{I,B3}} \leq 4000$

###### 3. Non-negativity

All $x_{p,e} \geq 0$

##### Retrieved Information

{
  "processing_time": {
    "A1": {"I": 5, "II": 10},
    "A2": {"I": 7, "II": 9, "III": 12},
    "B1": {"I": 6, "II": 8},
    "B2": {"I": 4, "III": 11},
    "B3": {"I": 7}
  },
  "available_time": {
    "A1": 6000,
    "A2": 10000,
    "B1": 4000,
    "B2": 7000,
    "B3": 4000
  },
  "equipment_cost": {
    "A1": 300,
    "A2": 321,
    "B1": 250,
    "B2": 783,
    "B3": 200
  },
  "raw_material_cost": {
    "I": 0.25,
    "II": 0.35,
    "III": 0.5
  },
  "unit_price": {
    "I": 1.25,
    "II": 2,
    "III": 2.8
  }
}

##### Variables

- $x_{\text{I,A1}}, x_{\text{I,A2}}, x_{\text{I,B1}}, x_{\text{I,B2}}, x_{\text{I,B3}} \geq 0$
- $x_{\text{II,A1}}, x_{\text{II,A2}}, x_{\text{II,B1}} \geq 0$
- $x_{\text{III,A2}}, x_{\text{III,B2}} \geq 0$

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & [1.25 - 0.25] x_{\text{I}} + [2 - 0.35] x_{\text{II}} + [2.8 - 0.5] x_{\text{III}} \\
& - (300 + 321 + 250 + 783 + 200) \\
\text{s.t.} \quad
& x_{\text{I}} = x_{\text{I,A1}} + x_{\text{I,A2}} = x_{\text{I,B1}} + x_{\text{I,B2}} + x_{\text{I,B3}} \\
& x_{\text{II}} = x_{\text{II,A1}} + x_{\text{II,A2}} = x_{\text{II,B1}} \\
& x_{\text{III}} = x_{\text{III,A2}} = x_{\text{III,B2}} \\
& 5 x_{\text{I,A1}} + 10 x_{\text{II,A1}} \leq 6000 \\
& 7 x_{\text{I,A2}} + 9 x_{\text{II,A2}} + 12 x_{\text{III,A2}} \leq 10000 \\
& 6 x_{\text{I,B1}} + 8 x_{\text{II,B1}} \leq 4000 \\
& 4 x_{\text{I,B2}} + 11 x_{\text{III,B2}} \leq 7000 \\
& 7 x_{\text{I,B3}} \leq 4000 \\
& x_{\text{I,A1}}, x_{\text{I,A2}}, x_{\text{I,B1}}, x_{\text{I,B2}}, x_{\text{I,B3}} \geq 0 \\
& x_{\text{II,A1}}, x_{\text{II,A2}}, x_{\text{II,B1}} \geq 0 \\
& x_{\text{III,A2}}, x_{\text{III,B2}} \geq 0
\end{align*}
\]

where

\[
\begin{align*}
x_{\text{I}} &= x_{\text{I,A1}} + x_{\text{I,A2}} = x_{\text{I,B1}} + x_{\text{I,B2}} + x_{\text{I,B3}} \\
x_{\text{II}} &= x_{\text{II,A1}} + x_{\text{II,A2}} = x_{\text{II,B1}} \\
x_{\text{III}} &= x_{\text{III,A2}} = x_{\text{III,B2}}
\end{align*}
\]

All variables are continuous and non-negative.