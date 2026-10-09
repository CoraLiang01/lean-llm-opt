Let x_i be the number of units of product i to order each day, for each product i in the order listed below. All x_i are nonnegative integers.

Define the following indices and parameters (in the order supplied):

| i | ProductName           | Weight | Value |
|---|-----------------------|--------|-------|
| 1 | Spinach               | 230    | 64    |
| 2 | Shiitake Mushrooms    | 637    | 75    |
| 3 | Apples                | 773    | 68    |
| 4 | Carrots               | 653    | 11    |
| 5 | Basil                 | 755    | 91    |
| 6 | Potatoes              | 670    | 31    |
| 7 | Green Beans           | 505    | 90    |
| 8 | Blueberries           | 821    | 56    |
| 9 | Oranges               | 83     | 10    |
|10 | Watermelons           | 249    | 24    |

Let Capacity = 875.

Decision variables:
x_i ∈ {0, 1, 2, ...} for i = 1,...,10

Mathematical Model:

Maximize total benefit:
 maximize 64 x₁ + 75 x₂ + 68 x₃ + 11 x₄ + 91 x₅ + 31 x₆ + 90 x₇ + 56 x₈ + 10 x₉ + 24 x₁₀

Subject to the stock capacity constraint:
 230 x₁ + 637 x₂ + 773 x₃ + 653 x₄ + 755 x₅ + 670 x₆ + 505 x₇ + 821 x₈ + 83 x₉ + 249 x₁₀ ≤ 875

and integrality and nonnegativity:
 x_i ∈ {0, 1, 2, ...} for all i = 1,...,10

Where:
- x₁ = units of Spinach to order
- x₂ = units of Shiitake Mushrooms to order
- x₃ = units of Apples to order
- x₄ = units of Carrots to order
- x₅ = units of Basil to order
- x₆ = units of Potatoes to order
- x₇ = units of Green Beans to order
- x₈ = units of Blueberries to order
- x₉ = units of Oranges to order
- x₁₀ = units of Watermelons to order

All coefficients and product order are as supplied in the data.