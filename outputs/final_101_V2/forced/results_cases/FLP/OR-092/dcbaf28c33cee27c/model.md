##### Decision Variables

- $x_{ij} \geq 0$: Amount of cargo (units) shipped from source $i$ to destination $j$ (continuous).
- $t_{ij} \in \mathbb{Z}_{\geq 0}$: Number of trucks dispatched from source $i$ to destination $j$ (integer, nonnegative).

Where $i \in I = \{\text{S1}, \text{S2}, ..., \text{S10}\}$ and $j \in J = \{\text{D1}, \text{D2}, ..., \text{D20}\}$.

##### Parameters

- Supply at each source:
  - S1: 103
  - S2: 87
  - S3: 95
  - S4: 112
  - S5: 97
  - S6: 103
  - S7: 101
  - S8: 94
  - S9: 102
  - S10: 106

- Demand at each destination:
  - D1: 61
  - D2: 54
  - D3: 56
  - D4: 54
  - D5: 53
  - D6: 47
  - D7: 56
  - D8: 57
  - D9: 56
  - D10: 34
  - D11: 55
  - D12: 53
  - D13: 37
  - D14: 31
  - D15: 62
  - D16: 58
  - D17: 39
  - D18: 32
  - D19: 38
  - D20: 67

- Cost per unit shipped from each source to each destination ($c_{ij}$):

|       | D1   | D2   | D3   | D4   | D5   | D6   | D7   | D8   | D9   | D10  | D11  | D12  | D13  | D14  | D15  | D16  | D17  | D18  | D19  | D20  |
|-------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| S1    | 2.75 | 2.6  | 2.9  | 1.7  | 1.85 | 1.98 | 2.74 | 6.2  | 5.75 | 6.44 | 5.2  | 4.54 | 5.39 | 4.34 | 8.28 | 8.87 | 9.03 | 8.49 | 9.66 | 10.91 |
| S2    | 5.24 | 4.87 | 4.72 | 4.12 | 4.26 | 4.55 | 4.45 | 3.88 | 3.24 | 4.24 | 3.13 | 3.16 | 3.67 | 2.43 | 5.83 | 6.51 | 6.68 | 6.1  | 7.2  | 8.4   |
| S3    | 5.54 | 5.16 | 5.09 | 4.37 | 4.57 | 4.78 | 4.89 | 3.84 | 3.08 | 4.42 | 3.37 | 3.53 | 3.98 | 2.7  | 5.84 | 6.48 | 6.68 | 6.14 | 7.24 | 8.31  |
| S4    | 4.51 | 4.17 | 4.01 | 3.36 | 3.59 | 3.73 | 3.68 | 4.39 | 3.93 | 4.71 | 3.55 | 3.15 | 3.84 | 2.6  | 6.51 | 7.03 | 7.27 | 6.69 | 7.89 | 9.04  |
| S5    | 4.84 | 4.74 | 4.77 | 3.8  | 3.96 | 4.1  | 4.52 | 4.82 | 4.03 | 5.35 | 4.25 | 4.14 | 4.73 | 3.46 | 6.89 | 7.5  | 7.67 | 7.2  | 8.2  | 9.35  |
| S6    | 9.13 | 8.32 | 7.78 | 7.9  | 8.13 | 8.45 | 7.47 | 3.04 | 4.25 | 2.19 | 3.34 | 4.07 | 3.3  | 4.24 | 2.77 | 2.63 | 2.97 | 2.56 | 3.32 | 5.09  |
| S7    | 8.64 | 7.86 | 7.34 | 7.32 | 7.51 | 7.83 | 7.03 | 1.9  | 3.14 | 1.43 | 2.64 | 3.64 | 2.92 | 3.51 | 2.41 | 2.74 | 2.98 | 2.46 | 3.51 | 5.08  |
| S8    | 9.25 | 8.41 | 7.81 | 8.13 | 8.32 | 8.6  | 7.49 | 3.71 | 5.0  | 2.88 | 3.78 | 4.27 | 3.49 | 4.63 | 3.62 | 3.34 | 3.59 | 3.29 | 3.79 | 5.54  |
| S9    | 10.3 | 9.58 | 8.89 | 9.11 | 9.32 | 9.69 | 8.57 | 4.11 | 5.29 | 3.49 | 4.59 | 5.28 | 4.47 | 5.51 | 3.22 | 2.59 | 2.72 | 2.68 | 2.82 | 4.47  |
| S10   | 7.85 | 7.08 | 6.57 | 6.72 | 6.88 | 7.18 | 6.21 | 2.34 | 3.56 | 1.52 | 2.18 | 2.78 | 2.01 | 3.0  | 3.47 | 3.62 | 3.89 | 3.41 | 4.38 | 5.98  |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Supply constraints (at each source):**
   \[
   \sum_{j \in J} x_{ij} \leq \text{supply}_i, \quad \forall i \in I
   \]
   where $\text{supply}_i$ is as listed above.

2. **Demand constraints (at each destination):**
   \[
   \sum_{i \in I} x_{ij} = \text{demand}_j, \quad \forall j \in J
   \]
   where $\text{demand}_j$ is as listed above.

3. **Truck loading constraints (per route):**
   \[
   x_{ij} \leq 10 \cdot t_{ij}, \quad \forall i \in I,\, j \in J
   \]
   (Each truck can carry up to 10 units; $t_{ij}$ is the integer number of trucks dispatched.)

4. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad t_{ij} \in \mathbb{Z}_{\geq 0}
   \]

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}, \text{D6}, \text{D7}, \text{D8}, \text{D9}, \text{D10}, \text{D11}, \text{D12}, \text{D13}, \text{D14}, \text{D15}, \text{D16}, \text{D17}, \text{D18}, \text{D19}, \text{D20}\}$

##### Summary

- All supply, demand, and cost data are as retrieved above.
- Each $x_{ij}$ is the amount shipped from $i$ to $j$ (continuous, nonnegative).
- Each $t_{ij}$ is the number of trucks dispatched from $i$ to $j$ (integer, nonnegative).
- $x_{ij} \leq 10 t_{ij}$ ensures no truck is overloaded and partial loading is allowed.
- The objective is to minimize total transportation cost, measured per unit shipped.