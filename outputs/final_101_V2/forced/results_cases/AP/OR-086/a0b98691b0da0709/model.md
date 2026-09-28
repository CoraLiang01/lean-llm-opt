##### Decision Variables

Let $x_{ij}$ denote the amount (in kg) of raw grade $i$ used in brand $j$, where:

- $i \in \{\text{I}, \text{II}, \text{III}\}$ (raw grades)
- $j \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$ (brands)

Let $y_j$ denote the total production (in kg) of brand $j$:
$$
y_j = \sum_{i} x_{ij} \quad \forall j \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
$$

##### Objective Function

Maximize total net profit (total sales revenue minus total raw material cost):

$$
\max \left\{ \sum_{j} p_j y_j - \sum_{i} c_i \sum_{j} x_{ij} \right\}
$$

where:

- $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$ (selling prices in CNY/kg)
- $c_{\text{I}} = 6$, $c_{\text{II}} = 4.5$, $c_{\text{III}} = 3$ (raw grade costs in CNY/kg)

##### Constraints

###### 1. Blending Requirements

- For Red: $\frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$, $\frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$
- For Yellow: $\frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$, $\frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$
- For Blue: $\frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$, $\frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$

Expressed as linear inequalities (for $y_j > 0$):

- Red:
  - $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$
  - $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$
- Yellow:
  - $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$
  - $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$
- Blue:
  - $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$
  - $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$

###### 2. Raw Material Supply Constraints

For each grade $i$:
$$
\sum_{j} x_{ij} \leq S_i
$$

where:

- $S_{\text{I}} = 1500$
- $S_{\text{II}} = 2000$
- $S_{\text{III}} = 1000$

###### 3. Minimum Production Constraint

$$
y_{\text{Red}} \geq 2000
$$

###### 4. Non-negativity

$$
x_{ij} \geq 0 \quad \forall i, j
$$

##### Retrieved Information

{
  "raw_materials": {
    "I": {"daily_supply": 1500, "cost": 6},
    "II": {"daily_supply": 2000, "cost": 4.5},
    "III": {"daily_supply": 1000, "cost": 3}
  },
  "brands": {
    "Red": {
      "blending_requirements": {
        "I": {"upper": 0.10},
        "II": {"lower": 0.50}
      },
      "selling_price": 5.5
    },
    "Yellow": {
      "blending_requirements": {
        "III": {"upper": 0.70},
        "I": {"lower": 0.20}
      },
      "selling_price": 5
    },
    "Blue": {
      "blending_requirements": {
        "I": {"upper": 0.50},
        "II": {"lower": 0.10}
      },
      "selling_price": 4.8
    }
  },
  "minimum_production": {
    "Red": 2000
  }
}