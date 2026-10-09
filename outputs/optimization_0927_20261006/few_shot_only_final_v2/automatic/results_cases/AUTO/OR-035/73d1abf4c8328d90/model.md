Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following breads in the order given in products.csv:

1. Baguette
2. Croissant
3. Sourdough
4. Rye Bread
5. Brioche
6. Focaccia
7. Ciabatta
8. Pita
9. Bagel
10. English Muffin

Parameters:
- Profits ($p_i$): 
  - Baguette: 888
  - Croissant: 134
  - Sourdough: 129
  - Rye Bread: 370
  - Brioche: 921
  - Focaccia: 765
  - Ciabatta: 154
  - Pita: 837
  - Bagel: 584
  - English Muffin: 365

- Weights ($w_i$): 
  - Baguette: 4
  - Croissant: 2
  - Sourdough: 4
  - Rye Bread: 3
  - Brioche: 2
  - Focaccia: 1
  - Ciabatta: 2
  - Pita: 1
  - Bagel: 3
  - English Muffin: 3

- Total storage capacity: 180

Mathematical Model:

Maximize
$$
888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
$$

Subject to
$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
$$