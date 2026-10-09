##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of bread type $i$ to order each day, for each $i \in \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}$.

##### Parameters

- Expected profit per unit ($v_i$):

  - Baguette: $v_{\text{Baguette}} = 888$
  - Croissant: $v_{\text{Croissant}} = 134$
  - Sourdough: $v_{\text{Sourdough}} = 129$
  - Rye Bread: $v_{\text{Rye Bread}} = 370$
  - Brioche: $v_{\text{Brioche}} = 921$
  - Focaccia: $v_{\text{Focaccia}} = 765$
  - Ciabatta: $v_{\text{Ciabatta}} = 154$
  - Pita: $v_{\text{Pita}} = 837$
  - Bagel: $v_{\text{Bagel}} = 584$
  - English Muffin: $v_{\text{English Muffin}} = 365$

- Storage weight per unit ($w_i$):

  - Baguette: $w_{\text{Baguette}} = 4$
  - Croissant: $w_{\text{Croissant}} = 2$
  - Sourdough: $w_{\text{Sourdough}} = 4$
  - Rye Bread: $w_{\text{Rye Bread}} = 3$
  - Brioche: $w_{\text{Brioche}} = 2$
  - Focaccia: $w_{\text{Focaccia}} = 1$
  - Ciabatta: $w_{\text{Ciabatta}} = 2$
  - Pita: $w_{\text{Pita}} = 1$
  - Bagel: $w_{\text{Bagel}} = 3$
  - English Muffin: $w_{\text{English Muffin}} = 3$

- Total storage capacity: $C = 180$

##### Objective Function

\[
\max \left(
888\,x_{\text{Baguette}} +
134\,x_{\text{Croissant}} +
129\,x_{\text{Sourdough}} +
370\,x_{\text{Rye Bread}} +
921\,x_{\text{Brioche}} +
765\,x_{\text{Focaccia}} +
154\,x_{\text{Ciabatta}} +
837\,x_{\text{Pita}} +
584\,x_{\text{Bagel}} +
365\,x_{\text{English Muffin}}
\right)
\]

##### Constraints

1. Storage capacity:
\[
4\,x_{\text{Baguette}} +
2\,x_{\text{Croissant}} +
4\,x_{\text{Sourdough}} +
3\,x_{\text{Rye Bread}} +
2\,x_{\text{Brioche}} +
1\,x_{\text{Focaccia}} +
2\,x_{\text{Ciabatta}} +
1\,x_{\text{Pita}} +
3\,x_{\text{Bagel}} +
3\,x_{\text{English Muffin}}
\leq 180
\]

2. Integer and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
\]

##### Retrieved Information

- Capacity: $C = 180$
- Products, values, and weights:

| Product           | Value | Weight |
|-------------------|-------|--------|
| Baguette          | 888   | 4      |
| Croissant         | 134   | 2      |
| Sourdough         | 129   | 4      |
| Rye Bread         | 370   | 3      |
| Brioche           | 921   | 2      |
| Focaccia          | 765   | 1      |
| Ciabatta          | 154   | 2      |
| Pita              | 837   | 1      |
| Bagel             | 584   | 3      |
| English Muffin    | 365   | 3      |