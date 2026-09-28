##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread type $i$ to order each day, for each $i \in I$ (where $I$ is the set of bread types).

##### Parameters

Let $I = \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}$.

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
\max \sum_{i \in I} v_i x_i
\]

##### Constraints

1. Storage capacity:
   \[
   \sum_{i \in I} w_i x_i \leq 180
   \]
2. Integer nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

##### Full Model (with all parameters)

\[
\begin{align*}
\max\quad & 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} \\
& + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}} \\
\text{s.t.}\quad & 4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} \\
& + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
\]

###### Retrieved Information

{
  "capacity": 180,
  "products": [
    {"ProductName": "Baguette", "Value": 888, "Weight": 4},
    {"ProductName": "Croissant", "Value": 134, "Weight": 2},
    {"ProductName": "Sourdough", "Value": 129, "Weight": 4},
    {"ProductName": "Rye Bread", "Value": 370, "Weight": 3},
    {"ProductName": "Brioche", "Value": 921, "Weight": 2},
    {"ProductName": "Focaccia", "Value": 765, "Weight": 1},
    {"ProductName": "Ciabatta", "Value": 154, "Weight": 2},
    {"ProductName": "Pita", "Value": 837, "Weight": 1},
    {"ProductName": "Bagel", "Value": 584, "Weight": 3},
    {"ProductName": "English Muffin", "Value": 365, "Weight": 3}
  ]
}