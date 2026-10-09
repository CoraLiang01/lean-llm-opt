##### Sets

- Products: $P = \{\text{I}, \text{II}, \text{III}\}$
- Procedure A equipment: $A = \{\text{A1}, \text{A2}\}$
- Procedure B equipment: $B = \{\text{B1}, \text{B2}, \text{B3}\}$

##### Parameters

- Processing time per unit (minutes/unit):

  - Procedure A:
    - $t_{\text{A1},\text{I}} = 5$, $t_{\text{A1},\text{II}} = 10$, $t_{\text{A1},\text{III}}$ undefined
    - $t_{\text{A2},\text{I}} = 7$, $t_{\text{A2},\text{II}} = 9$, $t_{\text{A2},\text{III}} = 12$
  - Procedure B:
    - $t_{\text{B1},\text{I}} = 6$, $t_{\text{B1},\text{II}} = 8$, $t_{\text{B1},\text{III}}$ undefined
    - $t_{\text{B2},\text{I}} = 4$, $t_{\text{B2},\text{II}}$ undefined, $t_{\text{B2},\text{III}} = 11$
    - $t_{\text{B3},\text{I}} = 7$, $t_{\text{B3},\text{II}}$ undefined, $t_{\text{B3},\text{III}}$ undefined

- Available equipment operating time (minutes):

  - $T_{\text{A1}} = 6000$
  - $T_{\text{A2}} = 10000$
  - $T_{\text{B1}} = 4000$
  - $T_{\text{B2}} = 7000$
  - $T_{\text{B3}} = 4000$

- Equipment cost at full load (yuan):

  - $C_{\text{A1}} = 300$
  - $C_{\text{A2}} = 321$
  - $C_{\text{B1}} = 250$
  - $C_{\text{B2}} = 783$
  - $C_{\text{B3}} = 200$

- Raw material cost per unit:
  - $r_{\text{I}} = 0.25$
  - $r_{\text{II}} = 0.35$
  - $r_{\text{III}} = 0.5$

- Selling price per unit:
  - $s_{\text{I}} = 1.25$
  - $s_{\text{II}} = 2$
  - $s_{\text{III}} = 2.8$

##### Decision Variables

- $x_{\text{A}k,p} \geq 0$: quantity of product $p$ processed on equipment $k$ for procedure A, for $k \in A$, $p \in P$
- $x_{\text{B}l,p} \geq 0$: quantity of product $p$ processed on equipment $l$ for procedure B, for $l \in B$, $p \in P$
- $y_p \geq 0$: total production quantity of product $p$, for $p \in P$

##### Eligibility Constraints

- Product I:
  - Procedure A: $x_{\text{A1},\text{I}},\ x_{\text{A2},\text{I}}$ allowed
  - Procedure B: $x_{\text{B1},\text{I}},\ x_{\text{B2},\text{I}},\ x_{\text{B3},\text{I}}$ allowed
- Product II:
  - Procedure A: $x_{\text{A1},\text{II}},\ x_{\text{A2},\text{II}}$ allowed
  - Procedure B: $x_{\text{B1},\text{II}}$ only
- Product III:
  - Procedure A: $x_{\text{A2},\text{III}}$ only
  - Procedure B: $x_{\text{B2},\text{III}}$ only

All other $x_{k,p}$ not listed above are fixed to zero.

##### Objective Function

Maximize total profit:
\[
\max \left[
\sum_{p \in P} s_p y_p
- \sum_{p \in P} r_p y_p
- \sum_{k \in A} \frac{C_k}{T_k} \sum_{p \in P} t_{k,p} x_{k,p}
- \sum_{l \in B} \frac{C_l}{T_l} \sum_{p \in P} t_{l,p} x_{l,p}
\right]
\]

##### Constraints

1. **Production balance for each product:**
   - For each $p \in P$:
     - $\sum_{k \in A} x_{k,p} = y_p$
     - $\sum_{l \in B} x_{l,p} = y_p$

2. **Equipment time limits:**
   - For each $k \in A$:
     - $\sum_{p \in P} t_{k,p} x_{k,p} \leq T_k$
   - For each $l \in B$:
     - $\sum_{p \in P} t_{l,p} x_{l,p} \leq T_l$

3. **Eligibility (enforced by variable definition):**
   - $x_{k,p} = 0$ if product $p$ cannot be processed on equipment $k$.

4. **Nonnegativity:**
   - $x_{k,p} \geq 0$, $x_{l,p} \geq 0$, $y_p \geq 0$

##### Explicit Variable List (nonzero only):

- $x_{\text{A1},\text{I}},\ x_{\text{A1},\text{II}}$
- $x_{\text{A2},\text{I}},\ x_{\text{A2},\text{II}},\ x_{\text{A2},\text{III}}$
- $x_{\text{B1},\text{I}},\ x_{\text{B1},\text{II}}$
- $x_{\text{B2},\text{I}},\ x_{\text{B2},\text{III}}$
- $x_{\text{B3},\text{I}}$

##### Data Table

| Equipment | Product I | Product II | Product III | Available Time | Full Load Cost |
|-----------|-----------|------------|-------------|---------------|---------------|
| A1        | 5         | 10         | —           | 6000          | 300           |
| A2        | 7         | 9          | 12          | 10000         | 321           |
| B1        | 6         | 8          | —           | 4000          | 250           |
| B2        | 4         | —          | 11          | 7000          | 783           |
| B3        | 7         | —          | —           | 4000          | 200           |

Raw material cost per unit: I: 0.25, II: 0.35, III: 0.5

Unit price per unit: I: 1.25, II: 2, III: 2.8

##### Complete Mathematical Model

Let $x_{\text{A1},\text{I}},\ x_{\text{A1},\text{II}},\ x_{\text{A2},\text{I}},\ x_{\text{A2},\text{II}},\ x_{\text{A2},\text{III}},\ x_{\text{B1},\text{I}},\ x_{\text{B1},\text{II}},\ x_{\text{B2},\text{I}},\ x_{\text{B2},\text{III}},\ x_{\text{B3},\text{I}} \geq 0$ and $y_{\text{I}},\ y_{\text{II}},\ y_{\text{III}} \geq 0$.

\[
\begin{align*}
\max\quad & 1.25 y_{\text{I}} + 2 y_{\text{II}} + 2.8 y_{\text{III}}
- 0.25 y_{\text{I}} - 0.35 y_{\text{II}} - 0.5 y_{\text{III}} \\
& - \left[ \frac{300}{6000}(5 x_{\text{A1},\text{I}} + 10 x_{\text{A1},\text{II}}) + \frac{321}{10000}(7 x_{\text{A2},\text{I}} + 9 x_{\text{A2},\text{II}} + 12 x_{\text{A2},\text{III}}) \right] \\
& - \left[ \frac{250}{4000}(6 x_{\text{B1},\text{I}} + 8 x_{\text{B1},\text{II}}) + \frac{783}{7000}(4 x_{\text{B2},\text{I}} + 11 x_{\text{B2},\text{III}}) + \frac{200}{4000}(7 x_{\text{B3},\text{I}}) \right] \\
\text{s.t.} \\
& x_{\text{A1},\text{I}} + x_{\text{A2},\text{I}} = y_{\text{I}} \\
& x_{\text{A1},\text{II}} + x_{\text{A2},\text{II}} = y_{\text{II}} \\
& x_{\text{A2},\text{III}} = y_{\text{III}} \\
& x_{\text{B1},\text{I}} + x_{\text{B2},\text{I}} + x_{\text{B3},\text{I}} = y_{\text{I}} \\
& x_{\text{B1},\text{II}} = y_{\text{II}} \\
& x_{\text{B2},\text{III}} = y_{\text{III}} \\
& 5 x_{\text{A1},\text{I}} + 10 x_{\text{A1},\text{II}} \leq 6000 \\
& 7 x_{\text{A2},\text{I}} + 9 x_{\text{A2},\text{II}} + 12 x_{\text{A2},\text{III}} \leq 10000 \\
& 6 x_{\text{B1},\text{I}} + 8 x_{\text{B1},\text{II}} \leq 4000 \\
& 4 x_{\text{B2},\text{I}} + 11 x_{\text{B2},\text{III}} \leq 7000 \\
& 7 x_{\text{B3},\text{I}} \leq 4000 \\
& x_{\text{A1},\text{I}},\ x_{\text{A1},\text{II}},\ x_{\text{A2},\text{I}},\ x_{\text{A2},\text{II}},\ x_{\text{A2},\text{III}},\ x_{\text{B1},\text{I}},\ x_{\text{B1},\text{II}},\ x_{\text{B2},\text{I}},\ x_{\text{B2},\text{III}},\ x_{\text{B3},\text{I}},\ y_{\text{I}},\ y_{\text{II}},\ y_{\text{III}} \geq 0
\end{align*}
\]

All variables not listed above are fixed to zero.

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
  "equipment_cost_full_load": {
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