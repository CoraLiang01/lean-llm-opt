##### Decision Variables

Let $x_i \in \mathbb{Z}_{\geq 0}$ be the number of units of bread type $i$ to order each day, for each $i$ in the set of products below.

##### Parameters

- Storage capacity: $180$
- For each bread type $i$:

| $i$                | Value $v_i$ | Weight $w_i$ |
|--------------------|-------------|--------------|
| Baguette           | 888         | 4            |
| Croissant          | 134         | 2            |
| Sourdough          | 129         | 4            |
| Rye Bread          | 370         | 3            |
| Brioche            | 921         | 2            |
| Focaccia           | 765         | 1            |
| Ciabatta           | 154         | 2            |
| Pita               | 837         | 1            |
| Bagel              | 584         | 3            |
| English Muffin     | 365         | 3            |

##### Objective Function

$\max\ 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}$

##### Constraints

$\begin{align*}
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} &\leq 180 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}$

##### Retrieved Information

- Storage capacity: $180$
- Products (in source order):

  - Baguette: Value $888$, Weight $4$
  - Croissant: Value $134$, Weight $2$
  - Sourdough: Value $129$, Weight $4$
  - Rye Bread: Value $370$, Weight $3$
  - Brioche: Value $921$, Weight $2$
  - Focaccia: Value $765$, Weight $1$
  - Ciabatta: Value $154$, Weight $2$
  - Pita: Value $837$, Weight $1$
  - Bagel: Value $584$, Weight $3$
  - English Muffin: Value $365$, Weight $3$