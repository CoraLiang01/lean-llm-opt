##### Decision Variables

Let $x_{ij}$ denote the amount (kg) of raw grade $i$ used in brand $j$, where:

- $i \in \{\text{I}, \text{II}, \text{III}\}$ (raw grades)
- $j \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$ (brands)

Let $y_j$ denote the total production (kg) of brand $j$:
$$
y_j = \sum_{i} x_{ij} \quad \forall j \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
$$

##### Objective Function

Maximize total net profit (total sales revenue minus total raw material cost):

$$
\max \Bigg[
\sum_{j \in \{\text{Red}, \text{Yellow}, \text{Blue}\}} p_j \cdot y_j
-
\sum_{i \in \{\text{I}, \text{II}, \text{III}\}} c_i \cdot \sum_{j} x_{ij}
\Bigg]
$$

where:

- $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$ (CNY/kg)
- $c_{\text{I}} = 6$, $c_{\text{II}} = 4.5$, $c_{\text{III}} = 3$ (CNY/kg)

##### Constraints

###### 1. Blending Requirements

- For Red:
  - Proportion of I less than 10%: $\dfrac{x_{\text{I,Red}}}{y_{\text{Red}}} < 0.10 \implies x_{\text{I,Red}} \leq 0.10\, y_{\text{Red}}$
  - Proportion of II more than 50%: $\dfrac{x_{\text{II,Red}}}{y_{\text{Red}}} > 0.50 \implies x_{\text{II,Red}} \geq 0.50\, y_{\text{Red}}$

- For Yellow:
  - Proportion of III less than 70%: $\dfrac{x_{\text{III,Yellow}}}{y_{\text{Yellow}}} < 0.70 \implies x_{\text{III,Yellow}} \leq 0.70\, y_{\text{Yellow}}$
  - Proportion of I more than 20%: $\dfrac{x_{\text{I,Yellow}}}{y_{\text{Yellow}}} > 0.20 \implies x_{\text{I,Yellow}} \geq 0.20\, y_{\text{Yellow}}$

- For Blue:
  - Proportion of I less than 50%: $\dfrac{x_{\text{I,Blue}}}{y_{\text{Blue}}} < 0.50 \implies x_{\text{I,Blue}} \leq 0.50\, y_{\text{Blue}}$
  - Proportion of II more than 10%: $\dfrac{x_{\text{II,Blue}}}{y_{\text{Blue}}} > 0.10 \implies x_{\text{II,Blue}} \geq 0.10\, y_{\text{Blue}}$

###### 2. Raw Material Supply Constraints

- For each grade $i$:
  $$
  \sum_{j} x_{ij} \leq S_i
  $$
  where $S_{\text{I}} = 1500$, $S_{\text{II}} = 2000$, $S_{\text{III}} = 1000$ (kg)

###### 3. Minimum Production Constraint

- Red brand must have at least 2,000 kg produced:
  $$
  y_{\text{Red}} \geq 2000
  $$

###### 4. Non-negativity

$$
x_{ij} \geq 0 \quad \forall i, j
$$

##### Retrieved Information

{
  "raw_grades": {
    "I": {
      "Daily Supply (kg)": 1500,
      "Cost (CNY/kg)": 6
    },
    "II": {
      "Daily Supply (kg)": 2000,
      "Cost (CNY/kg)": 4.5
    },
    "III": {
      "Daily Supply (kg)": 1000,
      "Cost (CNY/kg)": 3
    }
  },
  "brands": {
    "Red": {
      "Selling Price (CNY/kg)": 5.5,
      "Blending Requirements": [
        {"grade": "I", "type": "upper", "bound": 0.10},
        {"grade": "II", "type": "lower", "bound": 0.50}
      ]
    },
    "Yellow": {
      "Selling Price (CNY/kg)": 5,
      "Blending Requirements": [
        {"grade": "III", "type": "upper", "bound": 0.70},
        {"grade": "I", "type": "lower", "bound": 0.20}
      ]
    },
    "Blue": {
      "Selling Price (CNY/kg)": 4.8,
      "Blending Requirements": [
        {"grade": "I", "type": "upper", "bound": 0.50},
        {"grade": "II", "type": "lower", "bound": 0.10}
      ]
    }
  }
}

##### Full Model Summary

**Variables:** $x_{ij} \geq 0$ (kg of grade $i$ in brand $j$), $y_j = \sum_{i} x_{ij}$

**Objective:**
$$
\max \left[
5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}}
- 6 \sum_{j} x_{\text{I},j}
- 4.5 \sum_{j} x_{\text{II},j}
- 3 \sum_{j} x_{\text{III},j}
\right]
$$

**Subject to:**
- $x_{\text{I,Red}} \leq 0.10\, y_{\text{Red}}$
- $x_{\text{II,Red}} \geq 0.50\, y_{\text{Red}}$
- $x_{\text{III,Yellow}} \leq 0.70\, y_{\text{Yellow}}$
- $x_{\text{I,Yellow}} \geq 0.20\, y_{\text{Yellow}}$
- $x_{\text{I,Blue}} \leq 0.50\, y_{\text{Blue}}$
- $x_{\text{II,Blue}} \geq 0.10\, y_{\text{Blue}}$
- $\sum_{j} x_{\text{I},j} \leq 1500$
- $\sum_{j} x_{\text{II},j} \leq 2000$
- $\sum_{j} x_{\text{III},j} \leq 1000$
- $y_{\text{Red}} \geq 2000$
- $x_{ij} \geq 0$ for all $i, j$