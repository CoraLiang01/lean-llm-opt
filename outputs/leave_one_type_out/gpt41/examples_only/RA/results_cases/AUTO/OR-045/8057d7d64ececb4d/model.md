Let the index i run over the 10 produce types in the order given in products.csv:

1: Spinach  
2: Shiitake Mushrooms  
3: Apples  
4: Carrots  
5: Basil  
6: Potatoes  
7: Green Beans  
8: Blueberries  
9: Oranges  
10: Watermelons  

Let x_i = number of units of produce type i to order daily (integer, x_i ≥ 0).

Parameters (from products.csv):

| i  | ProductName          | Weight (w_i) | Value (v_i) |
|----|----------------------|--------------|-------------|
| 1  | Spinach              | 282          | 49          |
| 2  | Shiitake Mushrooms   | 83           | 30          |
| 3  | Apples               | 251          | 30          |
| 4  | Carrots              | 257          | 18          |
| 5  | Basil                | 88           | 54          |
| 6  | Potatoes             | 52           | 27          |
| 7  | Green Beans          | 198          | 91          |
| 8  | Blueberries          | 203          | 88          |
| 9  | Oranges              | 87           | 78          |
| 10 | Watermelons          | 265          | 22          |

Capacity (from capacity.csv):  
Total inventory weight capacity = 1035

Mathematical Model:

Decision variables:
x_i ∈ {0, 1, 2, ...} for i = 1,...,10

Objective:
Maximize total benefit:
maximize  
 49 x₁ + 30 x₂ + 30 x₃ + 18 x₄ + 54 x₅ + 27 x₆ + 91 x₇ + 88 x₈ + 78 x₉ + 22 x₁₀

Subject to:
Total weight constraint:
 282 x₁ + 83 x₂ + 251 x₃ + 257 x₄ + 88 x₅ + 52 x₆ + 198 x₇ + 203 x₈ + 87 x₉ + 265 x₁₀ ≤ 1035

x_i ∈ {0, 1, 2, ...} for all i = 1,...,10

Where:
x₁ = Spinach, x₂ = Shiitake Mushrooms, x₃ = Apples, x₄ = Carrots, x₅ = Basil, x₆ = Potatoes, x₇ = Green Beans, x₈ = Blueberries, x₉ = Oranges, x₁₀ = Watermelons.