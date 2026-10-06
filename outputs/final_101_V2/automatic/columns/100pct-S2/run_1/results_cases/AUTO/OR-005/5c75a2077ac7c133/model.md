Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as given by item_name in "products.csv".

Objective:
$$
\max \left( 888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}} \right)
$$

Subject to:

Storage capacity constraint:
$$
4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180
$$

Integrality constraints:
$$
x_{\text{Baguette}},\ x_{\text{Croissant}},\ x_{\text{Sourdough}},\ x_{\text{Rye Bread}},\ x_{\text{Brioche}},\ x_{\text{Focaccia}},\ x_{\text{Ciabatta}},\ x_{\text{Pita}},\ x_{\text{Bagel}},\ x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
$$

Where:

- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- item_value = expected profit per unit (from "products.csv")
- resource_requirement = storage space required per unit (from "products.csv")
- resource_capacity = total available storage space per day (180, from "capacity.csv")