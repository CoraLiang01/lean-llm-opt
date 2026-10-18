##### Sets and Indices

- Products: $P = \{\text{I}, \text{II}, \text{III}\}$
- Procedure A Equipment: $A = \{\text{A1}, \text{A2}\}$
- Procedure B Equipment: $B = \{\text{B1}, \text{B2}, \text{B3}\}$

##### Parameters (from 43.csv)

- Processing time per unit of product $p$ on equipment $e$: $t_{e,p}$ (minutes/unit)
- Available operating time for equipment $e$: $T_e$ (minutes)
- Equipment cost at full load for equipment $e$: $C_e$ (yuan)
- Raw material cost per unit of product $p$: $r_p$ (yuan/unit)
- Selling price per unit of product $p$: $s_p$ (yuan/unit)

###### Data

- **Processing times and equipment data:**

| Equipment | Product I | Product II | Product III | Available Time $T_e$ | Full Load Cost $C_e$ |
|-----------|-----------|------------|-------------|----------------------|----------------------|
| A1        | 5         | 10         | —           | 6000                 | 300                  |
| A2        | 7         | 9          | 12          | 10000                | 321                  |
| B1        | 6         | 8          | —           | 4000                 | 250                  |
| B2        | 4         | —          | 11          | 7000                 | 783                  |
| B3        | 7         | —          | —           | 4000                 | 200                  |

- **Raw material cost:**

| Product   | I    | II   | III  |
|-----------|------|------|------|
| $r_p$     | 0.25 | 0.35 | 0.5  |

- **Selling price:**

| Product   | I    | II   | III  |
|-----------|------|------|------|
| $s_p$     | 1.25 | 2    | 2.8  |

##### Decision Variables

- $x_{A,e,p} \geq 0$: Amount of product $p$ processed on procedure A equipment $e$.
- $x_{B,e,p} \geq 0$: Amount of product $p$ processed on procedure B equipment $e$.

##### Eligibility Constraints

- **Procedure A:**
  - Product I: $e \in \{\text{A1}, \text{A2}\}$
  - Product II: $e \in \{\text{A1}, \text{A2}\}$
  - Product III: $e = \text{A2}$

- **Procedure B:**
  - Product I: $e \in \{\text{B1}, \text{B2}, \text{B3}\}$
  - Product II: $e = \text{B1}$
  - Product III: $e = \text{B2}$

##### Objective Function

Maximize total profit:

\[
\max \left[
\sum_{p \in P} s_p \cdot y_p
- \sum_{p \in P} r_p \cdot y_p
- \sum_{e} C_e \cdot \left( \frac{1}{T_e} \sum_{p} t_{e,p} \cdot x_{e,p} \right)
\right]
\]

where $y_p$ is the total production of product $p$ (see below).

##### Constraints

1. **Production Consistency:**
   - For each product $p$:
     - The total amount processed in procedure A equals the total amount processed in procedure B:
       \[
       \sum_{e \in A_p} x_{A,e,p} = \sum_{e \in B_p} x_{B,e,p} = y_p
       \]
       where $A_p$ and $B_p$ are the eligible equipment sets for product $p$ in procedures A and B, respectively.

2. **Equipment Time Limits:**
   - For each equipment $e$:
     \[
     \sum_{p} t_{e,p} \cdot x_{e,p} \leq T_e
     \]
     where $x_{e,p}$ is $x_{A,e,p}$ if $e$ is an A-type equipment, $x_{B,e,p}$ if $e$ is a B-type equipment, and $t_{e,p}$ is defined only for eligible $(e,p)$ pairs.

3. **Eligibility Mask:**
   - For all ineligible $(e,p)$ pairs, set $x_{e,p} = 0$.

##### Variable Domains

\[
x_{A,e,p} \geq 0, \quad x_{B,e,p} \geq 0, \quad y_p \geq 0
\]

##### Explicit Model with Data

Let:

- $A = \{\text{A1}, \text{A2}\}$
- $B = \{\text{B1}, \text{B2}, \text{B3}\}$

- $A_{\text{I}} = \{\text{A1}, \text{A2}\}$, $B_{\text{I}} = \{\text{B1}, \text{B2}, \text{B3}\}$
- $A_{\text{II}} = \{\text{A1}, \text{A2}\}$, $B_{\text{II}} = \{\text{B1}\}$
- $A_{\text{III}} = \{\text{A2}\}$, $B_{\text{III}} = \{\text{B2}\}$

Define variables:

- $x_{A1,\text{I}},\ x_{A2,\text{I}},\ x_{A1,\text{II}},\ x_{A2,\text{II}},\ x_{A2,\text{III}}$
- $x_{B1,\text{I}},\ x_{B2,\text{I}},\ x_{B3,\text{I}},\ x_{B1,\text{II}},\ x_{B2,\text{III}}$
- $y_{\text{I}},\ y_{\text{II}},\ y_{\text{III}}$

**Objective:**

\[
\max \Bigg[
1.25\,y_{\text{I}} + 2\,y_{\text{II}} + 2.8\,y_{\text{III}}
- 0.25\,y_{\text{I}} - 0.35\,y_{\text{II}} - 0.5\,y_{\text{III}}
\]
\[
- 300 \cdot \frac{1}{6000}(5\,x_{A1,\text{I}} + 10\,x_{A1,\text{II}})
- 321 \cdot \frac{1}{10000}(7\,x_{A2,\text{I}} + 9\,x_{A2,\text{II}} + 12\,x_{A2,\text{III}})
\]
\[
- 250 \cdot \frac{1}{4000}(6\,x_{B1,\text{I}} + 8\,x_{B1,\text{II}})
- 783 \cdot \frac{1}{7000}(4\,x_{B2,\text{I}} + 11\,x_{B2,\text{III}})
- 200 \cdot \frac{1}{4000}(7\,x_{B3,\text{I}})
\Bigg]
\]

**Subject to:**

- **Production consistency:**
  - $x_{A1,\text{I}} + x_{A2,\text{I}} = x_{B1,\text{I}} + x_{B2,\text{I}} + x_{B3,\text{I}} = y_{\text{I}}$
  - $x_{A1,\text{II}} + x_{A2,\text{II}} = x_{B1,\text{II}} = y_{\text{II}}$
  - $x_{A2,\text{III}} = x_{B2,\text{III}} = y_{\text{III}}$

- **Equipment time limits:**
  - $5\,x_{A1,\text{I}} + 10\,x_{A1,\text{II}} \leq 6000$
  - $7\,x_{A2,\text{I}} + 9\,x_{A2,\text{II}} + 12\,x_{A2,\text{III}} \leq 10000$
  - $6\,x_{B1,\text{I}} + 8\,x_{B1,\text{II}} \leq 4000$
  - $4\,x_{B2,\text{I}} + 11\,x_{B2,\text{III}} \leq 7000$
  - $7\,x_{B3,\text{I}} \leq 4000$

- **Eligibility mask:**
  - All variables not listed above are fixed to zero.

- **Nonnegativity:**
  - All $x_{e,p} \geq 0$, all $y_p \geq 0$

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
  "selling_price": {
    "I": 1.25,
    "II": 2,
    "III": 2.8
  }
}

##### Variable List

- $x_{A1,\text{I}},\ x_{A2,\text{I}},\ x_{A1,\text{II}},\ x_{A2,\text{II}},\ x_{A2,\text{III}}$
- $x_{B1,\text{I}},\ x_{B2,\text{I}},\ x_{B3,\text{I}},\ x_{B1,\text{II}},\ x_{B2,\text{III}}$
- $y_{\text{I}},\ y_{\text{II}},\ y_{\text{III}}$

All other $x_{e,p}$ are fixed to zero by eligibility.

##### Summary

This model maximizes profit (revenue minus raw material and equipment costs), subject to production consistency, equipment time limits, and eligibility constraints, using all data from 43.csv. All production variables are continuous and nonnegative.