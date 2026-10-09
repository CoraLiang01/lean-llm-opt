Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following bread types:

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

Let $p_i$ be the expected profit per unit of bread type $i$ (from "item_value"), and $a_i$ be the storage space required per unit of bread type $i$ (from "resource_requirement"). The total available storage capacity is $180$ (from "resource_capacity" in capacity.csv).

The mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & 888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} + 921\,x_{\text{Brioche}} \\
& + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}} \\
\text{subject to} \quad & 4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} + 2\,x_{\text{Brioche}} \\
& + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}
\end{align*}
$$

Where:

- $p_{\text{Baguette}} = 888$, $a_{\text{Baguette}} = 4$
- $p_{\text{Croissant}} = 134$, $a_{\text{Croissant}} = 2$
- $p_{\text{Sourdough}} = 129$, $a_{\text{Sourdough}} = 4$
- $p_{\text{Rye Bread}} = 370$, $a_{\text{Rye Bread}} = 3$
- $p_{\text{Brioche}} = 921$, $a_{\text{Brioche}} = 2$
- $p_{\text{Focaccia}} = 765$, $a_{\text{Focaccia}} = 1$
- $p_{\text{Ciabatta}} = 154$, $a_{\text{Ciabatta}} = 2$
- $p_{\text{Pita}} = 837$, $a_{\text{Pita}} = 1$
- $p_{\text{Bagel}} = 584$, $a_{\text{Bagel}} = 3$
- $p_{\text{English Muffin}} = 365$, $a_{\text{English Muffin}} = 3$

and the total storage capacity is $180$ units. All $x_i$ are nonnegative integers.