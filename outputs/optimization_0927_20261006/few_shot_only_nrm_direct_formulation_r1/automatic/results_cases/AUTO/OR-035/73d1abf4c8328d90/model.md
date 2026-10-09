**Parameters:**

Let $I$ be the set of bread types:
- Baguette
- Croissant
- Sourdough
- Rye Bread
- Brioche
- Focaccia
- Ciabatta
- Pita
- Bagel
- English Muffin

Let $p_i$ = Value of bread type $i$ (from products.csv)  
Let $w_i$ = Weight (storage requirement) of bread type $i$ (from products.csv)  
Let $C$ = 180 (total storage capacity, from capacity.csv)

**Decision Variables:**

$x_i$ = number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

**Mathematical Model:**

Maximize total expected profit:
$$
\max \quad 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to the storage capacity constraint:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

And integrality/nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Where:
- $p_{\text{Baguette}} = 888$, $w_{\text{Baguette}} = 4$
- $p_{\text{Croissant}} = 134$, $w_{\text{Croissant}} = 2$
- $p_{\text{Sourdough}} = 129$, $w_{\text{Sourdough}} = 4$
- $p_{\text{Rye Bread}} = 370$, $w_{\text{Rye Bread}} = 3$
- $p_{\text{Brioche}} = 921$, $w_{\text{Brioche}} = 2$
- $p_{\text{Focaccia}} = 765$, $w_{\text{Focaccia}} = 1$
- $p_{\text{Ciabatta}} = 154$, $w_{\text{Ciabatta}} = 2$
- $p_{\text{Pita}} = 837$, $w_{\text{Pita}} = 1$
- $p_{\text{Bagel}} = 584$, $w_{\text{Bagel}} = 3$
- $p_{\text{English Muffin}} = 365$, $w_{\text{English Muffin}} = 3$

**Complete Model:**

$$
\begin{align*}
\max \quad & 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} \\
& + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}} \\
\text{s.t.} \quad & 4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} \\
& + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
$$