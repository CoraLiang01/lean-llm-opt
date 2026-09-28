##### Decision Variables

Let $x_i$ denote the production quantity of component $i$, for $i = 1, 2, ..., 111$ (corresponding to components $C1, C2, ..., C111$).

##### Parameters

- $p_i$: unit price of component $i$ (from unit_price.csv)
- $a_{ij}$: unit processing time of component $i$ in workshop $j$ (from processing_time_unit.csv)
- $b_j$: total available working hours in workshop $j$ (from total_working_hours.csv)

Workshops:  
$j = 1$: Casting  
$j = 2$: Milling  
$j = 3$: Finishing  
$j = 4$: Assembly  
$j = 5$: QA & Packaging

##### Objective Function

$\max \sum_{i=1}^{111} p_i x_i$

##### Constraints

For each workshop $j$ $(j = 1, ..., 5)$:
$$
\sum_{i=1}^{111} a_{ij} x_i \leq b_j
$$

For all $i = 1, ..., 111$:
$$
x_i \geq 0 \quad \text{and integer}
$$

##### Retrieved Information

{
  "processing_time_unit": {
    "C1":     {"Casting": 0.74, "Milling": 0.6, "Finishing": 0.0, "Assembly": 4.84, "QA & Packaging": 0.92},
    "C2":     {"Casting": 0.77, "Milling": 3.38, "Finishing": 4.15, "Assembly": 0.0, "QA & Packaging": 0.0},
    "C3":     {"Casting": 1.41, "Milling": 0.0, "Finishing": 0.0, "Assembly": 3.8, "QA & Packaging": 1.08},
    "C4":     {"Casting": 2.11, "Milling": 0.0, "Finishing": 2.45, "Assembly": 2.01, "QA & Packaging": 3.39},
    "C5":     {"Casting": 2.19, "Milling": 1.25, "Finishing": 2.49, "Assembly": 3.03, "QA & Packaging": 2.54},
    "C6":     {"Casting": 1.31, "Milling": 2.66, "Finishing": 1.93, "Assembly": 2.25, "QA & Packaging": 0.0},
    "C7":     {"Casting": 3.75, "Milling": 4.56, "Finishing": 0.72, "Assembly": 0.0, "QA & Packaging": 3.44},
    "C8":     {"Casting": 3.47, "Milling": 4.81, "Finishing": 0.0, "Assembly": 0.95, "QA & Packaging": 3.84},
    "C9":     {"Casting": 3.8, "Milling": 3.83, "Finishing": 1.12, "Assembly": 2.11, "QA & Packaging": 2.45},
    "C10":    {"Casting": 3.81, "Milling": 2.29, "Finishing": 2.95, "Assembly": 1.19, "QA & Packaging": 4.99},
    ...
    "C111":   {"Casting": 3.81, "Milling": 2.67, "Finishing": 4.07, "Assembly": 4.02, "QA & Packaging": 3.21}
  },
  "unit_price": {
    "C1": 193, "C2": 64, "C3": 103, "C4": 210, "C5": 85, "C6": 126, "C7": 226, "C8": 94, "C9": 73, "C10": 120, ..., "C111": 142
  },
  "total_working_hours": {
    "Casting": 7650,
    "Milling": 6320,
    "Finishing": 5538,
    "Assembly": 5957,
    "QA & Packaging": 6988
  }
}

##### Full Mathematical Model

Let $I = \{C1, C2, ..., C111\}$, $J = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$.

$\max \sum_{i \in I} p_i x_i$

Subject to:

$\sum_{i \in I} a_{ij} x_i \leq b_j \quad \forall j \in J$

$x_i \geq 0$ and integer, $\forall i \in I$

Where all $p_i$, $a_{ij}$, and $b_j$ are as listed above and in the retrieved data.