##### Decision Variables

Let:
- $x_1$ = production quantity of Product I (continuous, $\geq 0$)
- $x_2$ = production quantity of Product II (continuous, $\geq 0$)
- $x_3$ = production quantity of Product III (continuous, $\geq 0$)

Let $y_{e}$ denote the fraction of full load used on equipment $e$ (continuous, $0 \leq y_{e} \leq 1$), for each equipment $e$.

##### Parameters

- Processing time per unit (minutes/unit) for each product on each equipment (blank means not allowed):

| Equipment | Product I | Product II | Product III |
|-----------|-----------|------------|-------------|
| A1        | 5         | 10         | —           |
| A2        | 7         | 9          | 12          |
| A3        | 6         | 11         | 2           |
| B1        | 6         | 8          | —           |
| B2        | 4         | —          | 11          |
| B3        | 7         | —          | —           |
| B4        | 3         | 5          | 8           |

- Available equipment operating time (minutes):

| Equipment | Available Time |
|-----------|---------------|
| A1        | 6000          |
| A2        | 10000         |
| A3        | 8000          |
| B1        | 4000          |
| B2        | 7000          |
| B3        | 4000          |
| B4        | 5000          |

- Equipment cost at full load (yuan):

| Equipment | Full Load Cost |
|-----------|---------------|
| A1        | 300           |
| A2        | 321           |
| A3        | 203           |
| B1        | 250           |
| B2        | 783           |
| B3        | 200           |
| B4        | 300           |

- Raw material cost per unit:

| Product I | Product II | Product III |
|-----------|------------|-------------|
| 0.25      | 0.35       | 0.5         |

- Unit price per product:

| Product I | Product II | Product III |
|-----------|------------|-------------|
| 1.25      | 2          | 2.8         |

##### Model

###### Objective Function

Maximize total profit:

\[
\max \Bigg[
\underbrace{1.25 x_1 + 2 x_2 + 2.8 x_3}_{\text{Total Revenue}}
- \underbrace{0.25 x_1 + 0.35 x_2 + 0.5 x_3}_{\text{Raw Material Cost}}
- \underbrace{300 y_{A1} + 321 y_{A2} + 203 y_{A3} + 250 y_{B1} + 783 y_{B2} + 200 y_{B3} + 300 y_{B4}}_{\text{Equipment Cost}}
\Bigg]
\]

###### Constraints

**1. Equipment time constraints (for each equipment, total assigned time cannot exceed available time):**

For each equipment $e$:
\[
\sum_{j} (\text{processing time of product } j \text{ on } e) \cdot x_j \leq (\text{available time of } e) \cdot y_e
\]
where the sum is over products $j$ that can be processed on $e$.

Explicitly:

- A1: $5x_1 + 10x_2 \leq 6000 y_{A1}$
- A2: $7x_1 + 9x_2 + 12x_3 \leq 10000 y_{A2}$
- A3: $6x_1 + 11x_2 + 2x_3 \leq 8000 y_{A3}$
- B1: $6x_1 + 8x_2 \leq 4000 y_{B1}$
- B2: $4x_1 + 11x_3 \leq 7000 y_{B2}$
- B3: $7x_1 \leq 4000 y_{B3}$
- B4: $3x_1 + 5x_2 + 8x_3 \leq 5000 y_{B4}$

**2. Processing assignment constraints (each product must be fully processed for both procedures, using allowed equipment):**

- For procedure A (sum of assigned times for each product $j$ over allowed A equipment must equal total units produced times 1 unit):

  - Product I: $x_1 = x_{1,A1} + x_{1,A2}$
  - Product II: $x_2 = x_{2,A1} + x_{2,A2} + x_{2,A3}$
  - Product III: $x_3 = x_{3,A2}$

  But since all $x_j$ are assigned directly, and the time constraints above ensure only allowed assignments, we can model $x_j$ directly, as above.

- For procedure B (similarly):

  - Product I: $x_1 = x_{1,B1} + x_{1,B2} + x_{1,B3}$
  - Product II: $x_2 = x_{2,B1}$
  - Product III: $x_3 = x_{3,B2}$

  Again, since only allowed assignments are possible, and the time constraints above ensure only allowed assignments, we can model $x_j$ directly.

**3. Non-negativity and bounds:**

\[
x_1 \geq 0,\quad x_2 \geq 0,\quad x_3 \geq 0
\]
\[
0 \leq y_{e} \leq 1 \quad \forall e \in \{A1, A2, A3, B1, B2, B3, B4\}
\]

##### Retrieved Information

{
  "processing_time": {
    "A1": {"Product I": 5, "Product II": 10},
    "A2": {"Product I": 7, "Product II": 9, "Product III": 12},
    "A3": {"Product I": 6, "Product II": 11, "Product III": 2},
    "B1": {"Product I": 6, "Product II": 8},
    "B2": {"Product I": 4, "Product III": 11},
    "B3": {"Product I": 7},
    "B4": {"Product I": 3, "Product II": 5, "Product III": 8}
  },
  "available_time": {
    "A1": 6000,
    "A2": 10000,
    "A3": 8000,
    "B1": 4000,
    "B2": 7000,
    "B3": 4000,
    "B4": 5000
  },
  "equipment_cost_full_load": {
    "A1": 300,
    "A2": 321,
    "A3": 203,
    "B1": 250,
    "B2": 783,
    "B3": 200,
    "B4": 300
  },
  "raw_material_cost": {
    "Product I": 0.25,
    "Product II": 0.35,
    "Product III": 0.5
  },
  "unit_price": {
    "Product I": 1.25,
    "Product II": 2,
    "Product III": 2.8
  },
  "procedure_constraints": {
    "Product I": {
      "A": ["A1", "A2"],
      "B": ["B1", "B2", "B3"]
    },
    "Product II": {
      "A": ["A1", "A2", "A3"],
      "B": ["B1"]
    },
    "Product III": {
      "A": ["A2"],
      "B": ["B2"]
    }
  }
}

##### Complete Mathematical Model

\[
\begin{align*}
\max\ & \Big[1.25 x_1 + 2 x_2 + 2.8 x_3 - 0.25 x_1 - 0.35 x_2 - 0.5 x_3 \\
&\quad - 300 y_{A1} - 321 y_{A2} - 203 y_{A3} - 250 y_{B1} - 783 y_{B2} - 200 y_{B3} - 300 y_{B4}\Big] \\
\text{s.t.}\quad
& 5x_1 + 10x_2 \leq 6000 y_{A1} \\
& 7x_1 + 9x_2 + 12x_3 \leq 10000 y_{A2} \\
& 6x_1 + 11x_2 + 2x_3 \leq 8000 y_{A3} \\
& 6x_1 + 8x_2 \leq 4000 y_{B1} \\
& 4x_1 + 11x_3 \leq 7000 y_{B2} \\
& 7x_1 \leq 4000 y_{B3} \\
& 3x_1 + 5x_2 + 8x_3 \leq 5000 y_{B4} \\
& x_1 \geq 0,\ x_2 \geq 0,\ x_3 \geq 0 \\
& 0 \leq y_{A1} \leq 1,\ 0 \leq y_{A2} \leq 1,\ 0 \leq y_{A3} \leq 1 \\
& 0 \leq y_{B1} \leq 1,\ 0 \leq y_{B2} \leq 1,\ 0 \leq y_{B3} \leq 1,\ 0 \leq y_{B4} \leq 1
\end{align*}
\]

All parameters, vectors, and matrices are included as retrieved.