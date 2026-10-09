##### Sets and Indices

Let $I = \{1,2,3,4,5,6,7,8\}$ be the set of section IDs (from "capacity.csv").

Let $J = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of product names (from "products.csv").

##### Parameters

Section capacities (from "capacity.csv"):
\[
\begin{align*}
\text{Capacity}_1 &= 100 \\
\text{Capacity}_2 &= 150 \\
\text{Capacity}_3 &= 120 \\
\text{Capacity}_4 &= 130 \\
\text{Capacity}_5 &= 90 \\
\text{Capacity}_6 &= 110 \\
\text{Capacity}_7 &= 160 \\
\text{Capacity}_8 &= 140 \\
\end{align*}
\]

Product values and weights (from "products.csv"):
\[
\begin{array}{cccc}
\text{Product} & \text{Value}_j & \text{Weight}_j \\
1 & 10 & 2 \\
2 & 15 & 3 \\
3 & 8 & 1 \\
4 & 12 & 2 \\
5 & 20 & 4 \\
6 & 25 & 5 \\
7 & 5 & 1 \\
8 & 30 & 6 \\
9 & 18 & 3 \\
10 & 22 & 4 \\
\end{array}
\]

##### Decision Variables

For each section $i \in I$ and product $j \in J$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ to stock in section $i$.

##### Objective Function

\[
\max \sum_{i=1}^8 \sum_{j=1}^{10} \text{Value}_j \cdot x_{ij}
\]
That is,
\[
\max \left(
\sum_{i=1}^8 \left[
10x_{i1} + 15x_{i2} + 8x_{i3} + 12x_{i4} + 20x_{i5} + 25x_{i6} + 5x_{i7} + 30x_{i8} + 18x_{i9} + 22x_{i10}
\right]
\right)
\]

##### Constraints

For each section $i \in I$:
\[
\sum_{j=1}^{10} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
That is, for each section:

- Section 1: $2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 4x_{1,5} + 5x_{1,6} + 1x_{1,7} + 6x_{1,8} + 3x_{1,9} + 4x_{1,10} \leq 100$
- Section 2: $2x_{2,1} + 3x_{2,2} + 1x_{2,3} + 2x_{2,4} + 4x_{2,5} + 5x_{2,6} + 1x_{2,7} + 6x_{2,8} + 3x_{2,9} + 4x_{2,10} \leq 150$
- Section 3: $2x_{3,1} + 3x_{3,2} + 1x_{3,3} + 2x_{3,4} + 4x_{3,5} + 5x_{3,6} + 1x_{3,7} + 6x_{3,8} + 3x_{3,9} + 4x_{3,10} \leq 120$
- Section 4: $2x_{4,1} + 3x_{4,2} + 1x_{4,3} + 2x_{4,4} + 4x_{4,5} + 5x_{4,6} + 1x_{4,7} + 6x_{4,8} + 3x_{4,9} + 4x_{4,10} \leq 130$
- Section 5: $2x_{5,1} + 3x_{5,2} + 1x_{5,3} + 2x_{5,4} + 4x_{5,5} + 5x_{5,6} + 1x_{5,7} + 6x_{5,8} + 3x_{5,9} + 4x_{5,10} \leq 90$
- Section 6: $2x_{6,1} + 3x_{6,2} + 1x_{6,3} + 2x_{6,4} + 4x_{6,5} + 5x_{6,6} + 1x_{6,7} + 6x_{6,8} + 3x_{6,9} + 4x_{6,10} \leq 110$
- Section 7: $2x_{7,1} + 3x_{7,2} + 1x_{7,3} + 2x_{7,4} + 4x_{7,5} + 5x_{7,6} + 1x_{7,7} + 6x_{7,8} + 3x_{7,9} + 4x_{7,10} \leq 160$
- Section 8: $2x_{8,1} + 3x_{8,2} + 1x_{8,3} + 2x_{8,4} + 4x_{8,5} + 5x_{8,6} + 1x_{8,7} + 6x_{8,8} + 3x_{8,9} + 4x_{8,10} \leq 140$

##### Variable Domains

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

##### Retrieved Information

Sections and capacities:
- SectionID 1: Capacity 100
- SectionID 2: Capacity 150
- SectionID 3: Capacity 120
- SectionID 4: Capacity 130
- SectionID 5: Capacity 90
- SectionID 6: Capacity 110
- SectionID 7: Capacity 160
- SectionID 8: Capacity 140

Products:
- Product 1: Value 10, Weight 2
- Product 2: Value 15, Weight 3
- Product 3: Value 8, Weight 1
- Product 4: Value 12, Weight 2
- Product 5: Value 20, Weight 4
- Product 6: Value 25, Weight 5
- Product 7: Value 5, Weight 1
- Product 8: Value 30, Weight 6
- Product 9: Value 18, Weight 3
- Product 10: Value 22, Weight 4