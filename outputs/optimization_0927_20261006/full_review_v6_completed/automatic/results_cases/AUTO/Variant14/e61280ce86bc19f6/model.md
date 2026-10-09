##### Sets

- $I = \{B1, B2, B3, B4, B5, B6, B7, B8\}$: set of candidate depots (Centers)
- $J = \{Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10\}$: set of service zones

##### Parameters

- Opening costs:
  - $c_{B1} = 11$
  - $c_{B2} = 14$
  - $c_{B3} = 10$
  - $c_{B4} = 13$
  - $c_{B5} = 16$
  - $c_{B6} = 9$
  - $c_{B7} = 12$
  - $c_{B8} = 15$

- Coverage sets (depots covering each zone):

  - $S_{Z1} = \{B1, B4\}$
  - $S_{Z2} = \{B1, B2\}$
  - $S_{Z3} = \{B2, B5\}$
  - $S_{Z4} = \{B3, B7\}$
  - $S_{Z5} = \{B1, B3, B8\}$
  - $S_{Z6} = \{B2, B4, B8\}$
  - $S_{Z7} = \{B4, B5\}$
  - $S_{Z8} = \{B3, B6\}$
  - $S_{Z9} = \{B5, B6, B8\}$
  - $S_{Z10} = \{B6, B7\}$

##### Decision Variables

- $y_i \in \{0,1\}$ for all $i \in I$: $y_i = 1$ if depot $i$ is opened, $0$ otherwise.

##### Objective Function

\[
\min \sum_{i \in I} c_i y_i = 11y_{B1} + 14y_{B2} + 10y_{B3} + 13y_{B4} + 16y_{B5} + 9y_{B6} + 12y_{B7} + 15y_{B8}
\]

##### Constraints

- Coverage: For each $j \in J$, at least one depot covering $j$ must be opened.

\[
\begin{align*}
y_{B1} + y_{B4} &\geq 1 \quad &\text{(for } Z1) \\
y_{B1} + y_{B2} &\geq 1 \quad &\text{(for } Z2) \\
y_{B2} + y_{B5} &\geq 1 \quad &\text{(for } Z3) \\
y_{B3} + y_{B7} &\geq 1 \quad &\text{(for } Z4) \\
y_{B1} + y_{B3} + y_{B8} &\geq 1 \quad &\text{(for } Z5) \\
y_{B2} + y_{B4} + y_{B8} &\geq 1 \quad &\text{(for } Z6) \\
y_{B4} + y_{B5} &\geq 1 \quad &\text{(for } Z7) \\
y_{B3} + y_{B6} &\geq 1 \quad &\text{(for } Z8) \\
y_{B5} + y_{B6} + y_{B8} &\geq 1 \quad &\text{(for } Z9) \\
y_{B6} + y_{B7} &\geq 1 \quad &\text{(for } Z10)
\end{align*}
\]

- Binary restrictions:

\[
y_i \in \{0,1\} \quad \forall i \in I
\]

##### Retrieved Information

{
  "depots": [
    {"Center": "B1", "OpeningCost": 11, "CoveredDistricts": ["Z1", "Z2", "Z5"]},
    {"Center": "B2", "OpeningCost": 14, "CoveredDistricts": ["Z2", "Z3", "Z6"]},
    {"Center": "B3", "OpeningCost": 10, "CoveredDistricts": ["Z4", "Z5", "Z8"]},
    {"Center": "B4", "OpeningCost": 13, "CoveredDistricts": ["Z1", "Z6", "Z7"]},
    {"Center": "B5", "OpeningCost": 16, "CoveredDistricts": ["Z3", "Z7", "Z9"]},
    {"Center": "B6", "OpeningCost": 9, "CoveredDistricts": ["Z8", "Z9", "Z10"]},
    {"Center": "B7", "OpeningCost": 12, "CoveredDistricts": ["Z4", "Z10"]},
    {"Center": "B8", "OpeningCost": 15, "CoveredDistricts": ["Z5", "Z6", "Z9"]}
  ],
  "service_zones": ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10"]
}