##### Decision Variables

$y_i \in \{0,1\}$: $=1$ if service center $i$ is opened, $0$ otherwise, for each $i \in I$.

##### Parameters

- $I = \{\text{SC1}, \text{SC2}, \text{SC3}, \text{SC4}, \text{SC5}, \text{SC6}, \text{SC7}, \text{SC8}\}$: set of candidate service centers.
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}, \text{D6}, \text{D7}, \text{D8}, \text{D9}, \text{D10}\}$: set of districts to be covered.
- Opening costs:
  - $c_{\text{SC1}} = 12$
  - $c_{\text{SC2}} = 15$
  - $c_{\text{SC3}} = 18$
  - $c_{\text{SC4}} = 10$
  - $c_{\text{SC5}} = 14$
  - $c_{\text{SC6}} = 13$
  - $c_{\text{SC7}} = 16$
  - $c_{\text{SC8}} = 11$
- Coverage sets (districts covered by each center):
  - $\text{SC1}: \{\text{D1}, \text{D2}, \text{D4}\}$
  - $\text{SC2}: \{\text{D2}, \text{D3}, \text{D5}\}$
  - $\text{SC3}: \{\text{D4}, \text{D5}, \text{D6}\}$
  - $\text{SC4}: \{\text{D6}, \text{D7}\}$
  - $\text{SC5}: \{\text{D7}, \text{D8}, \text{D10}\}$
  - $\text{SC6}: \{\text{D8}, \text{D9}\}$
  - $\text{SC7}: \{\text{D1}, \text{D9}, \text{D10}\}$
  - $\text{SC8}: \{\text{D3}, \text{D4}, \text{D8}\}$

Let $a_{ij} = 1$ if center $i$ covers district $j$, $0$ otherwise. The coverage matrix $[a_{ij}]$ is:

|        | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| SC1    | 1  | 1  | 0  | 1  | 0  | 0  | 0  | 0  | 0  | 0   |
| SC2    | 0  | 1  | 1  | 0  | 1  | 0  | 0  | 0  | 0  | 0   |
| SC3    | 0  | 0  | 0  | 1  | 1  | 1  | 0  | 0  | 0  | 0   |
| SC4    | 0  | 0  | 0  | 0  | 0  | 1  | 1  | 0  | 0  | 0   |
| SC5    | 0  | 0  | 0  | 0  | 0  | 0  | 1  | 1  | 0  | 1   |
| SC6    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 1  | 1  | 0   |
| SC7    | 1  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 1  | 1   |
| SC8    | 0  | 0  | 1  | 1  | 0  | 0  | 0  | 1  | 0  | 0   |

##### Objective Function

\[
\min \sum_{i \in I} c_i y_i = 12y_{\text{SC1}} + 15y_{\text{SC2}} + 18y_{\text{SC3}} + 10y_{\text{SC4}} + 14y_{\text{SC5}} + 13y_{\text{SC6}} + 16y_{\text{SC7}} + 11y_{\text{SC8}}
\]

##### Constraints

1. **Coverage:** Every district must be covered by at least one opened center:
   - For each $j \in J$:
     \[
     \sum_{i \in I} a_{ij} y_i \geq 1
     \]
   Explicitly, for each district:
   - D1: $y_{\text{SC1}} + y_{\text{SC7}} \geq 1$
   - D2: $y_{\text{SC1}} + y_{\text{SC2}} \geq 1$
   - D3: $y_{\text{SC2}} + y_{\text{SC8}} \geq 1$
   - D4: $y_{\text{SC1}} + y_{\text{SC3}} + y_{\text{SC8}} \geq 1$
   - D5: $y_{\text{SC2}} + y_{\text{SC3}} \geq 1$
   - D6: $y_{\text{SC3}} + y_{\text{SC4}} \geq 1$
   - D7: $y_{\text{SC4}} + y_{\text{SC5}} \geq 1$
   - D8: $y_{\text{SC5}} + y_{\text{SC6}} + y_{\text{SC8}} \geq 1$
   - D9: $y_{\text{SC6}} + y_{\text{SC7}} \geq 1$
   - D10: $y_{\text{SC5}} + y_{\text{SC7}} \geq 1$

2. **Binary restrictions:**
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\min\ & 12y_{\text{SC1}} + 15y_{\text{SC2}} + 18y_{\text{SC3}} + 10y_{\text{SC4}} + 14y_{\text{SC5}} + 13y_{\text{SC6}} + 16y_{\text{SC7}} + 11y_{\text{SC8}} \\
\text{s.t.}\quad
& y_{\text{SC1}} + y_{\text{SC7}} \geq 1 \\
& y_{\text{SC1}} + y_{\text{SC2}} \geq 1 \\
& y_{\text{SC2}} + y_{\text{SC8}} \geq 1 \\
& y_{\text{SC1}} + y_{\text{SC3}} + y_{\text{SC8}} \geq 1 \\
& y_{\text{SC2}} + y_{\text{SC3}} \geq 1 \\
& y_{\text{SC3}} + y_{\text{SC4}} \geq 1 \\
& y_{\text{SC4}} + y_{\text{SC5}} \geq 1 \\
& y_{\text{SC5}} + y_{\text{SC6}} + y_{\text{SC8}} \geq 1 \\
& y_{\text{SC6}} + y_{\text{SC7}} \geq 1 \\
& y_{\text{SC5}} + y_{\text{SC7}} \geq 1 \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]