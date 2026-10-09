##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether factory $i$ is constructed (binary).

##### Parameters

- $I = \{A1, A2, \ldots, A15\}$ (factories)
- $J = \{B1, B2, \ldots, B8\}$ (distribution centers)

- Factory fixed costs ($f_i$):

  | Factory | Fixed Cost |
  |---------|------------|
  | A1      | 0          |
  | A2      | 175        |
  | A3      | 300        |
  | A4      | 375        |
  | A5      | 500        |
  | A6      | 200        |
  | A7      | 260        |
  | A8      | 220        |
  | A9      | 320        |
  | A10     | 280        |
  | A11     | 350        |
  | A12     | 420        |
  | A13     | 470        |
  | A14     | 520        |
  | A15     | 560        |

- Factory capacities ($u_i$):

  | Factory | Capacity |
  |---------|----------|
  | A1      | 30       |
  | A2      | 10       |
  | A3      | 20       |
  | A4      | 30       |
  | A5      | 40       |
  | A6      | 20       |
  | A7      | 25       |
  | A8      | 30       |
  | A9      | 35       |
  | A10     | 20       |
  | A11     | 40       |
  | A12     | 25       |
  | A13     | 30       |
  | A14     | 50       |
  | A15     | 45       |

- Distribution center demands ($d_j$):

  | Distribution Center | Demand |
  |---------------------|--------|
  | B1                  | 30     |
  | B2                  | 25     |
  | B3                  | 20     |
  | B4                  | 35     |
  | B5                  | 25     |
  | B6                  | 30     |
  | B7                  | 25     |
  | B8                  | 30     |

- Shipping costs per unit ($c_{ij}$):

  |      | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
  |------|----|----|----|----|----|----|----|----|
  | A1   | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
  | A2   | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
  | A3   | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
  | A4   | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
  | A5   | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
  | A6   | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
  | A7   | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
  | A8   | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
  | A9   | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
  | A10  | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
  | A11  | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
  | A12  | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
  | A13  | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
  | A14  | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
  | A15  | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each distribution center $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Factory capacity (conditional on construction):**  
   For each factory $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Retrieved Information

```json
{
  "factories": [
    {"Facility": "A1", "FixedCost": "0", "Capacity": "30"},
    {"Facility": "A2", "FixedCost": "175", "Capacity": "10"},
    {"Facility": "A3", "FixedCost": "300", "Capacity": "20"},
    {"Facility": "A4", "FixedCost": "375", "Capacity": "30"},
    {"Facility": "A5", "FixedCost": "500", "Capacity": "40"},
    {"Facility": "A6", "FixedCost": "200", "Capacity": "20"},
    {"Facility": "A7", "FixedCost": "260", "Capacity": "25"},
    {"Facility": "A8", "FixedCost": "220", "Capacity": "30"},
    {"Facility": "A9", "FixedCost": "320", "Capacity": "35"},
    {"Facility": "A10", "FixedCost": "280", "Capacity": "20"},
    {"Facility": "A11", "FixedCost": "350", "Capacity": "40"},
    {"Facility": "A12", "FixedCost": "420", "Capacity": "25"},
    {"Facility": "A13", "FixedCost": "470", "Capacity": "30"},
    {"Facility": "A14", "FixedCost": "520", "Capacity": "50"},
    {"Facility": "A15", "FixedCost": "560", "Capacity": "45"}
  ],
  "distribution_centers": [
    {"Destination": "B1", "Demand": "30"},
    {"Destination": "B2", "Demand": "25"},
    {"Destination": "B3", "Demand": "20"},
    {"Destination": "B4", "Demand": "35"},
    {"Destination": "B5", "Demand": "25"},
    {"Destination": "B6", "Demand": "30"},
    {"Destination": "B7", "Demand": "25"},
    {"Destination": "B8", "Demand": "30"}
  ],
  "shipping_costs": {
    "A1": {"B1": "8", "B2": "4", "B3": "3", "B4": "6", "B5": "7", "B6": "5", "B7": "9", "B8": "8"},
    "A2": {"B1": "5", "B2": "2", "B3": "3", "B4": "5", "B5": "6", "B6": "4", "B7": "7", "B8": "6"},
    "A3": {"B1": "4", "B2": "3", "B3": "4", "B4": "6", "B5": "5", "B6": "5", "B7": "6", "B8": "7"},
    "A4": {"B1": "9", "B2": "7", "B3": "5", "B4": "8", "B5": "9", "B6": "6", "B7": "10", "B8": "7"},
    "A5": {"B1": "10", "B2": "4", "B3": "2", "B4": "6", "B5": "8", "B6": "5", "B7": "7", "B8": "3"},
    "A6": {"B1": "6", "B2": "5", "B3": "4", "B4": "5", "B5": "7", "B6": "6", "B7": "8", "B8": "5"},
    "A7": {"B1": "7", "B2": "6", "B3": "5", "B4": "4", "B5": "6", "B6": "7", "B7": "9", "B8": "6"},
    "A8": {"B1": "5", "B2": "4", "B3": "6", "B4": "3", "B5": "5", "B6": "6", "B7": "7", "B8": "6"},
    "A9": {"B1": "8", "B2": "7", "B3": "6", "B4": "7", "B5": "9", "B6": "8", "B7": "10", "B8": "7"},
    "A10": {"B1": "6", "B2": "5", "B3": "7", "B4": "4", "B5": "6", "B6": "5", "B7": "7", "B8": "5"},
    "A11": {"B1": "9", "B2": "6", "B3": "4", "B4": "6", "B5": "8", "B6": "7", "B7": "9", "B8": "6"},
    "A12": {"B1": "7", "B2": "5", "B3": "6", "B4": "5", "B5": "6", "B6": "5", "B7": "8", "B8": "5"},
    "A13": {"B1": "8", "B2": "6", "B3": "5", "B4": "6", "B5": "7", "B6": "6", "B7": "8", "B8": "7"},
    "A14": {"B1": "9", "B2": "5", "B3": "3", "B4": "5", "B5": "7", "B6": "4", "B7": "6", "B8": "4"},
    "A15": {"B1": "10", "B2": "6", "B3": "4", "B4": "5", "B5": "8", "B6": "5", "B7": "7", "B8": "5"}
  }
}
```