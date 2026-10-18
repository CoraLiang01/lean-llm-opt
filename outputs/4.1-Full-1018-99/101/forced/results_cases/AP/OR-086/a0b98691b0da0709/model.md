##### Decision Variables

Let $x_{g,b}$ denote the amount (kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $y_b = \sum_{g} x_{g,b}$ denote the total daily production (kg) of brand $b$.

##### Parameters

- Raw Grades: I, II, III
- Brands: Red, Yellow, Blue

- Raw Grade Daily Supply Limits and Costs:
  - Grade I: supply limit $S_{\text{I}} = 1500$ kg, cost $c_{\text{I}} = 6$ CNY/kg
  - Grade II: supply limit $S_{\text{II}} = 2000$ kg, cost $c_{\text{II}} = 4.5$ CNY/kg
  - Grade III: supply limit $S_{\text{III}} = 1000$ kg, cost $c_{\text{III}} = 3$ CNY/kg

- Brand Selling Prices:
  - Red: $p_{\text{Red}} = 5.5$ CNY/kg
  - Yellow: $p_{\text{Yellow}} = 5$ CNY/kg
  - Blue: $p_{\text{Blue}} = 4.8$ CNY/kg

- Blending Requirements:
  - Red: Grade I $< 10\%$, Grade II $> 50\%$
  - Yellow: Grade III $< 70\%$, Grade I $> 20\%$
  - Blue: Grade I $< 50\%$, Grade II $> 10\%$

##### Objective Function

$\max \left[ \sum_{b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}} p_b \cdot y_b - \sum_{g \in \{\text{I}, \text{II}, \text{III}\}} c_g \cdot \sum_{b} x_{g,b} \right]$

##### Constraints

###### 1. Blending Requirements

- For Red:
  - $\dfrac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (if $y_{\text{Red}} > 0$)
  - $\dfrac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (if $y_{\text{Red}} > 0$)

- For Yellow:
  - $\dfrac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (if $y_{\text{Yellow}} > 0$)
  - $\dfrac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (if $y_{\text{Yellow}} > 0$)

- For Blue:
  - $\dfrac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (if $y_{\text{Blue}} > 0$)
  - $\dfrac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (if $y_{\text{Blue}} > 0$)

Equivalently, for $y_b > 0$:
- $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$
- $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$
- $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$
- $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$
- $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$
- $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$

For $y_b = 0$, all $x_{g,b} = 0$.

###### 2. Raw Material Supply Constraints

$\sum_{b} x_{\text{I},b} \leq 1500$

$\sum_{b} x_{\text{II},b} \leq 2000$

$\sum_{b} x_{\text{III},b} \leq 1000$

###### 3. Minimum Production Constraint

$y_{\text{Red}} \geq 2000$

###### 4. Nonnegativity

$x_{g,b} \geq 0$ for all $g, b$

###### 5. Definition of $y_b$

$y_b = \sum_{g} x_{g,b}$ for all $b$

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
      "Blending Requirements": {
        "I": {"upper": 0.10},
        "II": {"lower": 0.50}
      },
      "Selling Price (CNY/kg)": 5.5
    },
    "Yellow": {
      "Blending Requirements": {
        "III": {"upper": 0.70},
        "I": {"lower": 0.20}
      },
      "Selling Price (CNY/kg)": 5
    },
    "Blue": {
      "Blending Requirements": {
        "I": {"upper": 0.50},
        "II": {"lower": 0.10}
      },
      "Selling Price (CNY/kg)": 4.8
    }
  }
}