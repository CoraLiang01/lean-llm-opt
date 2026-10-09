Let $x_i$ be the number of units of bread type $i$ to order each day. Each $x_i$ is a nonnegative integer.

Let the set of bread types $i$ and their parameters be:

\[
\begin{array}{llll}
\text{Bread Type} & \text{Expected Profit } (p_i) & \text{Space Requirement } (a_i) \\
\hline
\text{Baguette} & 888 & 4 \\
\text{Croissant} & 134 & 2 \\
\text{Sourdough} & 129 & 4 \\
\text{Rye Bread} & 370 & 3 \\
\text{Brioche} & 921 & 2 \\
\text{Focaccia} & 765 & 1 \\
\text{Ciabatta} & 154 & 2 \\
\text{Pita} & 837 & 1 \\
\text{Bagel} & 584 & 3 \\
\text{English Muffin} & 365 & 3 \\
\end{array}
\]

The total storage capacity is $180$ units.

The mathematical model is:

\[
\textbf{Objective:} \quad \max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
\]

\[
\textbf{Subject to:}
\]

\[
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}
\]