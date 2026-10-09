Let:
- $y_s \in \{0,1\}$: 1 if service centre $s$ is opened, 0 otherwise, for $s \in \{\text{SC1},\ldots,\text{SC10}\}$
- $x_{cs} \in \{0,1\}$: 1 if customer $c$ is assigned to centre $s$, 0 otherwise, for $c \in \{\text{C1},\ldots,\text{C15}\}$ and $s \in \{\text{SC1},\ldots,\text{SC10}\}$

Parameters:
- $f_s$: Fixed opening cost for centre $s$
- $c_{cs}$: Cost to serve customer $c$ from centre $s$

Data:

Fixed Opening Costs:
\[
\begin{array}{ll}
\text{SC1}: & 385.1 \\
\text{SC2}: & 546.3 \\
\text{SC3}: & 485.2 \\
\text{SC4}: & 448.1 \\
\text{SC5}: & 324.1 \\
\text{SC6}: & 323.9 \\
\text{SC7}: & 296.5 \\
\text{SC8}: & 522.7 \\
\text{SC9}: & 448.7 \\
\text{SC10}: & 478.7 \\
\end{array}
\]

Customer–Centre Service Costs ($c_{cs}$):

| Customer | SC1  | SC2  | SC3  | SC4  | SC5  | SC6  | SC7  | SC8  | SC9  | SC10 |
|----------|------|------|------|------|------|------|------|------|------|-------|
| C1       | 15.1 | 21.2 | 14.9 | 18.8 | 22.9 | 16.8 | 16.5 | 9.4  | 16.1 | 17.3  |
| C2       | 13.4 | 16.3 | 20.2 | 19.6 | 20.9 | 22.1 | 16.9 | 9.4  | 13.8 | 11.7  |
| C3       | 15.2 | 18.8 | 14.7 | 21.7 | 18.1 | 18.6 | 12.3 | 11.2 | 11.9 | 20.4  |
| C4       | 16.8 | 19.1 | 18.3 | 18.8 | 23.1 | 15.7 | 13.1 | 8.6  | 15.6 | 22.2  |
| C5       | 13.4 | 18.6 | 20.8 | 19.8 | 22.1 | 18.1 | 16.7 | 12.1 | 11.4 | 18.2  |
| C6       | 12.5 | 22.5 | 15.5 | 14.9 | 21.6 | 21.3 | 16.1 | 10.7 | 11.9 | 14.6  |
| C7       | 12.1 | 17.1 | 19.8 | 18.6 | 22.1 | 20.7 | 20.5 | 12.2 | 15.4 | 18.7  |
| C8       | 12.3 | 15.7 | 17.9 | 21.3 | 22.7 | 15.3 | 16.6 | 11.4 | 14.1 | 20.1  |
| C9       | 16.3 | 21.3 | 17.6 | 20.8 | 21.8 | 17.2 | 15.5 | 12.6 | 19.9 | 19.1  |
| C10      | 12.1 | 18.7 | 14.4 | 20.1 | 22.7 | 14.1 | 18.1 | 11.4 | 18.1 | 17.4  |
| C11      | 16.7 | 18.7 | 15.7 | 19.9 | 24.2 | 18.7 | 14.2 | 13.1 | 14.7 | 16.1  |
| C12      | 11.3 | 23.8 | 15.5 | 17.3 | 23.2 | 17.7 | 16.8 | 14.5 | 15.8 | 17.8  |
| C13      | 15.1 | 20.5 | 15.1 | 18.4 | 20.6 | 17.9 | 14.5 | 8.5  | 14.9 | 13.9  |
| C14      | 8.3  | 20.7 | 14.7 | 20.4 | 20.6 | 14.8 | 14.2 | 11.5 | 14.1 | 15.1  |
| C15      | 12.1 | 16.3 | 16.4 | 15.1 | 21.3 | 19.1 | 19.5 | 16.7 | 11.1 | 18.7  |

Model:

Minimise
\[
\sum_{s \in S} f_s y_s + \sum_{c \in C} \sum_{s \in S} c_{cs} x_{cs}
\]

Subject to

1. Each customer is assigned to exactly one centre:
\[
\sum_{s \in S} x_{cs} = 1 \qquad \forall c \in C
\]

2. Customers can only be assigned to open centres:
\[
x_{cs} \leq y_s \qquad \forall c \in C,\, s \in S
\]

3. Each open centre serves at most 4 customers:
\[
\sum_{c \in C} x_{cs} \leq 4 y_s \qquad \forall s \in S
\]

4. Variable domains:
\[
y_s \in \{0,1\} \qquad \forall s \in S
\]
\[
x_{cs} \in \{0,1\} \qquad \forall c \in C,\, s \in S
\]

Where:
- $S = \{\text{SC1},\ldots,\text{SC10}\}$
- $C = \{\text{C1},\ldots,\text{C15}\}$
- $f_s$ and $c_{cs}$ as given above.