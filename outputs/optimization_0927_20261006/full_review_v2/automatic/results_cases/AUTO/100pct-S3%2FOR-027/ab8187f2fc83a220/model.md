##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to centre $i \in I$, 0 otherwise.

##### Parameters

- $I = \{\text{SC1}, \text{SC2}, \text{SC3}, \text{SC4}, \text{SC5}, \text{SC6}, \text{SC7}, \text{SC8}, \text{SC9}, \text{SC10}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}, \text{C13}, \text{C14}, \text{C15}\}$

- Fixed opening costs $f_i$ for each centre $i$:

\[
\begin{align*}
f_{\text{SC1}} &= 385.1 \\
f_{\text{SC2}} &= 546.3 \\
f_{\text{SC3}} &= 485.2 \\
f_{\text{SC4}} &= 448.1 \\
f_{\text{SC5}} &= 324.1 \\
f_{\text{SC6}} &= 323.9 \\
f_{\text{SC7}} &= 296.5 \\
f_{\text{SC8}} &= 522.7 \\
f_{\text{SC9}} &= 448.7 \\
f_{\text{SC10}} &= 478.7 \\
\end{align*}
\]

- Service cost matrix $c_{ij}$ (cost to serve customer $j$ from centre $i$):

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

##### Objective Function

\[
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
\]

##### Constraints

1. **Assignment:** Each customer is assigned to exactly one centre:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Open centre only if assigned:** A customer can only be assigned to an open centre:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

3. **Centre capacity:** Each opened centre serves at most 4 customers:
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

###### Retrieved Information

```json
{
  "service_centers": [
    {"Service Center": "SC1", "Fixed Opening Cost": "385.1"},
    {"Service Center": "SC2", "Fixed Opening Cost": "546.3"},
    {"Service Center": "SC3", "Fixed Opening Cost": "485.2"},
    {"Service Center": "SC4", "Fixed Opening Cost": "448.1"},
    {"Service Center": "SC5", "Fixed Opening Cost": "324.1"},
    {"Service Center": "SC6", "Fixed Opening Cost": "323.9"},
    {"Service Center": "SC7", "Fixed Opening Cost": "296.5"},
    {"Service Center": "SC8", "Fixed Opening Cost": "522.7"},
    {"Service Center": "SC9", "Fixed Opening Cost": "448.7"},
    {"Service Center": "SC10", "Fixed Opening Cost": "478.7"}
  ],
  "customers": [
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15"
  ],
  "service_costs": {
    "C1":  {"SC1":15.1,"SC2":21.2,"SC3":14.9,"SC4":18.8,"SC5":22.9,"SC6":16.8,"SC7":16.5,"SC8":9.4,"SC9":16.1,"SC10":17.3},
    "C2":  {"SC1":13.4,"SC2":16.3,"SC3":20.2,"SC4":19.6,"SC5":20.9,"SC6":22.1,"SC7":16.9,"SC8":9.4,"SC9":13.8,"SC10":11.7},
    "C3":  {"SC1":15.2,"SC2":18.8,"SC3":14.7,"SC4":21.7,"SC5":18.1,"SC6":18.6,"SC7":12.3,"SC8":11.2,"SC9":11.9,"SC10":20.4},
    "C4":  {"SC1":16.8,"SC2":19.1,"SC3":18.3,"SC4":18.8,"SC5":23.1,"SC6":15.7,"SC7":13.1,"SC8":8.6,"SC9":15.6,"SC10":22.2},
    "C5":  {"SC1":13.4,"SC2":18.6,"SC3":20.8,"SC4":19.8,"SC5":22.1,"SC6":18.1,"SC7":16.7,"SC8":12.1,"SC9":11.4,"SC10":18.2},
    "C6":  {"SC1":12.5,"SC2":22.5,"SC3":15.5,"SC4":14.9,"SC5":21.6,"SC6":21.3,"SC7":16.1,"SC8":10.7,"SC9":11.9,"SC10":14.6},
    "C7":  {"SC1":12.1,"SC2":17.1,"SC3":19.8,"SC4":18.6,"SC5":22.1,"SC6":20.7,"SC7":20.5,"SC8":12.2,"SC9":15.4,"SC10":18.7},
    "C8":  {"SC1":12.3,"SC2":15.7,"SC3":17.9,"SC4":21.3,"SC5":22.7,"SC6":15.3,"SC7":16.6,"SC8":11.4,"SC9":14.1,"SC10":20.1},
    "C9":  {"SC1":16.3,"SC2":21.3,"SC3":17.6,"SC4":20.8,"SC5":21.8,"SC6":17.2,"SC7":15.5,"SC8":12.6,"SC9":19.9,"SC10":19.1},
    "C10": {"SC1":12.1,"SC2":18.7,"SC3":14.4,"SC4":20.1,"SC5":22.7,"SC6":14.1,"SC7":18.1,"SC8":11.4,"SC9":18.1,"SC10":17.4},
    "C11": {"SC1":16.7,"SC2":18.7,"SC3":15.7,"SC4":19.9,"SC5":24.2,"SC6":18.7,"SC7":14.2,"SC8":13.1,"SC9":14.7,"SC10":16.1},
    "C12": {"SC1":11.3,"SC2":23.8,"SC3":15.5,"SC4":17.3,"SC5":23.2,"SC6":17.7,"SC7":16.8,"SC8":14.5,"SC9":15.8,"SC10":17.8},
    "C13": {"SC1":15.1,"SC2":20.5,"SC3":15.1,"SC4":18.4,"SC5":20.6,"SC6":17.9,"SC7":14.5,"SC8":8.5,"SC9":14.9,"SC10":13.9},
    "C14": {"SC1":8.3,"SC2":20.7,"SC3":14.7,"SC4":20.4,"SC5":20.6,"SC6":14.8,"SC7":14.2,"SC8":11.5,"SC9":14.1,"SC10":15.1},
    "C15": {"SC1":12.1,"SC2":16.3,"SC3":16.4,"SC4":15.1,"SC5":21.3,"SC6":19.1,"SC7":19.5,"SC8":16.7,"SC9":11.1,"SC10":18.7}
  }
}
```