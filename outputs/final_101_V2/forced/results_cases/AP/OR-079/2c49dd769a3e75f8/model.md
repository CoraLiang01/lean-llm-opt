##### Objective Function:

$\quad \min \left[ \sum_{i=1}^{15} F_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{8} c_{ij} x_{ij} \right]$

where:
- $F_i$ is the fixed cost of opening factory $i$,
- $y_i$ is a binary variable indicating if factory $i$ is opened ($y_i \in \{0,1\}$),
- $c_{ij}$ is the per-unit shipping cost from factory $i$ to distribution center $j$,
- $x_{ij}$ is the quantity shipped from factory $i$ to distribution center $j$.

##### Constraints

###### 1. Demand Satisfaction at Distribution Centers:

$\sum_{i=1}^{15} x_{ij} = D_j \quad \forall j \in \{1,2,\ldots,8\}$

where $D_j$ is the demand at distribution center $j$.

###### 2. Factory Capacity and Activation:

$\sum_{j=1}^{8} x_{ij} \leq K_i y_i \quad \forall i \in \{1,2,\ldots,15\}$

where $K_i$ is the capacity of factory $i$.

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i$

$x_{ij} \geq 0 \quad \forall i, j$

---

##### Retrieved Information

```json
{
  "factories": [
    {"Facility": "A1", "FixedCost": 0, "Capacity": 30},
    {"Facility": "A2", "FixedCost": 175, "Capacity": 10},
    {"Facility": "A3", "FixedCost": 300, "Capacity": 20},
    {"Facility": "A4", "FixedCost": 375, "Capacity": 30},
    {"Facility": "A5", "FixedCost": 500, "Capacity": 40},
    {"Facility": "A6", "FixedCost": 200, "Capacity": 20},
    {"Facility": "A7", "FixedCost": 260, "Capacity": 25},
    {"Facility": "A8", "FixedCost": 220, "Capacity": 30},
    {"Facility": "A9", "FixedCost": 320, "Capacity": 35},
    {"Facility": "A10", "FixedCost": 280, "Capacity": 20},
    {"Facility": "A11", "FixedCost": 350, "Capacity": 40},
    {"Facility": "A12", "FixedCost": 420, "Capacity": 25},
    {"Facility": "A13", "FixedCost": 470, "Capacity": 30},
    {"Facility": "A14", "FixedCost": 520, "Capacity": 50},
    {"Facility": "A15", "FixedCost": 560, "Capacity": 45}
  ],
  "distribution_centers": [
    {"Destination": "B1", "Demand": 30},
    {"Destination": "B2", "Demand": 25},
    {"Destination": "B3", "Demand": 20},
    {"Destination": "B4", "Demand": 35},
    {"Destination": "B5", "Demand": 25},
    {"Destination": "B6", "Demand": 30},
    {"Destination": "B7", "Demand": 25},
    {"Destination": "B8", "Demand": 30}
  ],
  "shipping_costs": {
    "A1":  {"B1": 8,  "B2": 4,  "B3": 3,  "B4": 6,  "B5": 7,  "B6": 5,  "B7": 9,  "B8": 8},
    "A2":  {"B1": 5,  "B2": 2,  "B3": 3,  "B4": 5,  "B5": 6,  "B6": 4,  "B7": 7,  "B8": 6},
    "A3":  {"B1": 4,  "B2": 3,  "B3": 4,  "B4": 6,  "B5": 5,  "B6": 5,  "B7": 6,  "B8": 7},
    "A4":  {"B1": 9,  "B2": 7,  "B3": 5,  "B4": 8,  "B5": 9,  "B6": 6,  "B7": 10, "B8": 7},
    "A5":  {"B1": 10, "B2": 4,  "B3": 2,  "B4": 6,  "B5": 8,  "B6": 5,  "B7": 7,  "B8": 3},
    "A6":  {"B1": 6,  "B2": 5,  "B3": 4,  "B4": 5,  "B5": 7,  "B6": 6,  "B7": 8,  "B8": 5},
    "A7":  {"B1": 7,  "B2": 6,  "B3": 5,  "B4": 4,  "B5": 6,  "B6": 7,  "B7": 9,  "B8": 6},
    "A8":  {"B1": 5,  "B2": 4,  "B3": 6,  "B4": 3,  "B5": 5,  "B6": 6,  "B7": 7,  "B8": 6},
    "A9":  {"B1": 8,  "B2": 7,  "B3": 6,  "B4": 7,  "B5": 9,  "B6": 8,  "B7": 10, "B8": 7},
    "A10": {"B1": 6,  "B2": 5,  "B3": 7,  "B4": 4,  "B5": 6,  "B6": 5,  "B7": 7,  "B8": 5},
    "A11": {"B1": 9,  "B2": 6,  "B3": 4,  "B4": 6,  "B5": 8,  "B6": 7,  "B7": 9,  "B8": 6},
    "A12": {"B1": 7,  "B2": 5,  "B3": 6,  "B4": 5,  "B5": 6,  "B6": 5,  "B7": 8,  "B8": 5},
    "A13": {"B1": 8,  "B2": 6,  "B3": 5,  "B4": 6,  "B5": 7,  "B6": 6,  "B7": 8,  "B8": 7},
    "A14": {"B1": 9,  "B2": 5,  "B3": 3,  "B4": 5,  "B5": 7,  "B6": 4,  "B7": 6,  "B8": 4},
    "A15": {"B1": 10, "B2": 6,  "B3": 4,  "B4": 5,  "B5": 8,  "B6": 5,  "B7": 7,  "B8": 5}
  }
}
```

- Factories: A1–A15, with fixed costs and capacities as listed above.
- Distribution Centers: B1–B8, with demands as listed above.
- Shipping costs: $c_{ij}$ as per the table above, for all $i \in \{\text{A1},\ldots,\text{A15}\}$ and $j \in \{\text{B1},\ldots,\text{B8}\}$.

##### Variable and Parameter Definitions

- $y_i$: Binary variable, 1 if factory $i$ is built, 0 otherwise.
- $x_{ij}$: Amount shipped from factory $i$ to distribution center $j$.
- $F_i$: Fixed cost of factory $i$ (see table).
- $K_i$: Capacity of factory $i$ (see table).
- $D_j$: Demand at distribution center $j$ (see list).
- $c_{ij}$: Per-unit shipping cost from factory $i$ to distribution center $j$ (see table).

##### Sets

- $i \in \{\text{A1}, \text{A2}, \ldots, \text{A15}\}$
- $j \in \{\text{B1}, \text{B2}, \ldots, \text{B8}\}$

---

This model fully captures the facility location and shipment planning problem as described, using all provided data.