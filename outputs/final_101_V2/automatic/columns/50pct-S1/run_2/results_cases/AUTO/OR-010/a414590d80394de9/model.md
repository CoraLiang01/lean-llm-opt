Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

Indices:
- $i$ indexes SectionID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName from products.csv: $j \in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $c_i$ = Capacity of section $i$ (from capacity.csv)
- $p_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from products.csv)

Data:

From capacity.csv:
\[
\begin{array}{lll}
\text{SectionID} & \text{archive\_revision\_number} & \text{Capacity} \\
1 & 3 & 100 \\
2 & 7 & 150 \\
3 & 3 & 120 \\
4 & 4 & 130 \\
5 & 6 & 90 \\
6 & 4 & 110 \\
7 & 3 & 160 \\
8 & 8 & 140 \\
\end{array}
\]

From products.csv:
\[
\begin{array}{lllll}
\text{record\_keeper\_group} & \text{ProductName} & \text{Value} & \text{archive\_revision\_number} & \text{Weight} \\
\text{Team B} & 1 & 10 & 1 & 2 \\
\text{Team A} & 2 & 15 & 4 & 3 \\
\text{Team C} & 3 & 8 & 2 & 1 \\
\text{Team A} & 4 & 12 & 9 & 2 \\
\text{Team A} & 5 & 20 & 3 & 4 \\
\text{Team C} & 6 & 25 & 7 & 5 \\
\text{Team A} & 7 & 5 & 2 & 1 \\
\text{Team C} & 8 & 30 & 3 & 6 \\
\text{Team C} & 9 & 18 & 7 & 3 \\
\text{Team A} & 10 & 22 & 5 & 4 \\
\end{array}
\]

Model:

Objective:
\[
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} p_j x_{ij}
\]
where $p_j$ is the Value of product $j$.

Constraints:

For each section $i$ (SectionID from capacity.csv):
\[
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]
where $w_j$ is the Weight of product $j$, and $c_i$ is the Capacity of section $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

Explicitly, with the data:

Objective:
\[
\max \sum_{i=1}^{8} \Big( 10x_{i1} + 15x_{i2} + 8x_{i3} + 12x_{i4} + 20x_{i5} + 25x_{i6} + 5x_{i7} + 30x_{i8} + 18x_{i9} + 22x_{i10} \Big)
\]

For each section $i$:

Section 1 ($c_1=100$):
\[
2x_{11} + 3x_{12} + 1x_{13} + 2x_{14} + 4x_{15} + 5x_{16} + 1x_{17} + 6x_{18} + 3x_{19} + 4x_{1,10} \leq 100
\]

Section 2 ($c_2=150$):
\[
2x_{21} + 3x_{22} + 1x_{23} + 2x_{24} + 4x_{25} + 5x_{26} + 1x_{27} + 6x_{28} + 3x_{29} + 4x_{2,10} \leq 150
\]

Section 3 ($c_3=120$):
\[
2x_{31} + 3x_{32} + 1x_{33} + 2x_{34} + 4x_{35} + 5x_{36} + 1x_{37} + 6x_{38} + 3x_{39} + 4x_{3,10} \leq 120
\]

Section 4 ($c_4=130$):
\[
2x_{41} + 3x_{42} + 1x_{43} + 2x_{44} + 4x_{45} + 5x_{46} + 1x_{47} + 6x_{48} + 3x_{49} + 4x_{4,10} \leq 130
\]

Section 5 ($c_5=90$):
\[
2x_{51} + 3x_{52} + 1x_{53} + 2x_{54} + 4x_{55} + 5x_{56} + 1x_{57} + 6x_{58} + 3x_{59} + 4x_{5,10} \leq 90
\]

Section 6 ($c_6=110$):
\[
2x_{61} + 3x_{62} + 1x_{63} + 2x_{64} + 4x_{65} + 5x_{66} + 1x_{67} + 6x_{68} + 3x_{69} + 4x_{6,10} \leq 110
\]

Section 7 ($c_7=160$):
\[
2x_{71} + 3x_{72} + 1x_{73} + 2x_{74} + 4x_{75} + 5x_{76} + 1x_{77} + 6x_{78} + 3x_{79} + 4x_{7,10} \leq 160
\]

Section 8 ($c_8=140$):
\[
2x_{81} + 3x_{82} + 1x_{83} + 2x_{84} + 4x_{85} + 5x_{86} + 1x_{87} + 6x_{88} + 3x_{89} + 4x_{8,10} \leq 140
\]

And for all $i=1,\ldots,8$, $j=1,\ldots,10$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]