Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as given by the item_name column in products.csv. All $x_i$ are required to be nonnegative integers.

Objective:
\[
\max \sum_{i} \text{item\_value}_i \cdot x_i
\]
where $\text{item\_value}_i$ is the expected profit for bread type $i$.

Constraint (storage capacity):
\[
\sum_{i} \text{resource\_requirement}_i \cdot x_i \leq 180
\]
where $\text{resource\_requirement}_i$ is the storage requirement for bread type $i$, and $180$ is the total resource_capacity from capacity.csv.

Variable domains:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Numerical Formulation:

Let the bread types be indexed in the order of products.csv:

\[
\begin{align*}
\max\quad & 888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} \\
& + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} \\
& + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}} \\
\text{s.t.}\quad & 4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} \\
& + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} \\
& + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}
\end{align*}
\]

All coefficients and identifiers are taken directly from the retrieved data, in source order.