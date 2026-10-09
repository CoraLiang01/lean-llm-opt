##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of goods shipped from supplier $i$ to customer group $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- Demands $d_j$ for each customer $j$:
  - $d_{\text{C1}} = 216$
  - $d_{\text{C2}} = 168$
  - $d_{\text{C3}} = 264$
  - $d_{\text{C4}} = 216$
  - $d_{\text{C5}} = 216$
  - $d_{\text{C6}} = 192$
  - $d_{\text{C7}} = 144$
  - $d_{\text{C8}} = 168$
  - $d_{\text{C9}} = 168$
  - $d_{\text{C10}} = 168$

- Supply capacities $s_i$ for each supplier $i$:
  - $s_{\text{S1}} = 288$
  - $s_{\text{S2}} = 288$
  - $s_{\text{S3}} = 264$
  - $s_{\text{S4}} = 264$
  - $s_{\text{S5}} = 216$
  - $s_{\text{S6}} = 216$
  - $s_{\text{S7}} = 168$
  - $s_{\text{S8}} = 216$
  - $s_{\text{S9}} = 240$
  - $s_{\text{S10}} = 168$

- Transportation costs $c_{ij}$ (per unit from supplier $i$ to customer $j$):

|        | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 590.3648137  | 23.66917261  | 88.89005870  | 497.5222881  | 466.0903432  | 29.02209683  | 23.67524483  | 23.67776029  | 0.3118394915 | 58.89547392  |
| S2     | 2042.071500  | 2133.978484  | 705.1591203  | 101.5945452  | 2052.937657  | 1738.754951  | 101.6109497  | 101.6106217  | 122.4521427  | 67.29170751  |
| S3     | 22.29722216  | 497.9271939  | 1653.082886  | 23.68545123  | 1386.080789  | 26.13715281  | 497.6220483  | 498.0935847  | 865.3816296  | 1008.671739  |
| S4     | 960.7814534  | 49.12830005  | 1324.238697  | 1032.209548  | 0.078047254  | 53.30826873  | 49.13641672  | 1031.821442  | 466.0049531  | 1351.818907  |
| S5     | 1471.272167  | 85.69560728  | 38.89266824  | 1542.050036  | 112.2051437  | 82.37020164  | 1542.339920  | 85.69238746  | 1924.936077  | 1094.669596  |
| S6     | 191.9058726  | 158.5040103  | 91.02045350  | 184.4474720  | 968.1467987  | 284.1076062  | 8.791061588  | 158.7052384  | 27.94387435  | 929.8071683  |
| S7     | 81.23891457  | 0.3744642223 | 2079.466865  | 0.3065671756 | 1031.777296  | 7.203964492  | 0.0762307224 | 0.0324738795 | 23.68582797  | 849.9799407  |
| S8     | 56.09931097  | 935.6143109  | 73.08824617  | 52.00392409  | 4.025792389  | 1002.232766  | 935.7766030  | 935.7007252  | 612.8698719  | 1348.836615  |
| S9     | 4.502283327  | 0.3899585343 | 1782.466218  | 0.006345907  | 1031.991011  | 129.5066562  | 0.2118319573 | 0.6457301074 | 497.6272391  | 40.46575555  |
| S10    | 333.6869270  | 277.4719386  | 86.02096892  | 277.3083661  | 1004.464908  | 19.95033682  | 13.20207369  | 238.1432152  | 411.0580332  | 941.7526366  |

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

That is,
\[
\min \Bigg(
\begin{aligned}
&590.3648137\,x_{\text{S1},\text{C1}} + 23.66917261\,x_{\text{S1},\text{C2}} + 88.89005870\,x_{\text{S1},\text{C3}} + 497.5222881\,x_{\text{S1},\text{C4}} + 466.0903432\,x_{\text{S1},\text{C5}} \\
&+ 29.02209683\,x_{\text{S1},\text{C6}} + 23.67524483\,x_{\text{S1},\text{C7}} + 23.67776029\,x_{\text{S1},\text{C8}} + 0.3118394915\,x_{\text{S1},\text{C9}} + 58.89547392\,x_{\text{S1},\text{C10}} \\
&+ 2042.071500\,x_{\text{S2},\text{C1}} + 2133.978484\,x_{\text{S2},\text{C2}} + 705.1591203\,x_{\text{S2},\text{C3}} + 101.5945452\,x_{\text{S2},\text{C4}} + 2052.937657\,x_{\text{S2},\text{C5}} \\
&+ 1738.754951\,x_{\text{S2},\text{C6}} + 101.6109497\,x_{\text{S2},\text{C7}} + 101.6106217\,x_{\text{S2},\text{C8}} + 122.4521427\,x_{\text{S2},\text{C9}} + 67.29170751\,x_{\text{S2},\text{C10}} \\
&+ 22.29722216\,x_{\text{S3},\text{C1}} + 497.9271939\,x_{\text{S3},\text{C2}} + 1653.082886\,x_{\text{S3},\text{C3}} + 23.68545123\,x_{\text{S3},\text{C4}} + 1386.080789\,x_{\text{S3},\text{C5}} \\
&+ 26.13715281\,x_{\text{S3},\text{C6}} + 497.6220483\,x_{\text{S3},\text{C7}} + 498.0935847\,x_{\text{S3},\text{C8}} + 865.3816296\,x_{\text{S3},\text{C9}} + 1008.671739\,x_{\text{S3},\text{C10}} \\
&+ 960.7814534\,x_{\text{S4},\text{C1}} + 49.12830005\,x_{\text{S4},\text{C2}} + 1324.238697\,x_{\text{S4},\text{C3}} + 1032.209548\,x_{\text{S4},\text{C4}} + 0.078047254\,x_{\text{S4},\text{C5}} \\
&+ 53.30826873\,x_{\text{S4},\text{C6}} + 49.13641672\,x_{\text{S4},\text{C7}} + 1031.821442\,x_{\text{S4},\text{C8}} + 466.0049531\,x_{\text{S4},\text{C9}} + 1351.818907\,x_{\text{S4},\text{C10}} \\
&+ 1471.272167\,x_{\text{S5},\text{C1}} + 85.69560728\,x_{\text{S5},\text{C2}} + 38.89266824\,x_{\text{S5},\text{C3}} + 1542.050036\,x_{\text{S5},\text{C4}} + 112.2051437\,x_{\text{S5},\text{C5}} \\
&+ 82.37020164\,x_{\text{S5},\text{C6}} + 1542.339920\,x_{\text{S5},\text{C7}} + 85.69238746\,x_{\text{S5},\text{C8}} + 1924.936077\,x_{\text{S5},\text{C9}} + 1094.669596\,x_{\text{S5},\text{C10}} \\
&+ 191.9058726\,x_{\text{S6},\text{C1}} + 158.5040103\,x_{\text{S6},\text{C2}} + 91.02045350\,x_{\text{S6},\text{C3}} + 184.4474720\,x_{\text{S6},\text{C4}} + 968.1467987\,x_{\text{S6},\text{C5}} \\
&+ 284.1076062\,x_{\text{S6},\text{C6}} + 8.791061588\,x_{\text{S6},\text{C7}} + 158.7052384\,x_{\text{S6},\text{C8}} + 27.94387435\,x_{\text{S6},\text{C9}} + 929.8071683\,x_{\text{S6},\text{C10}} \\
&+ 81.23891457\,x_{\text{S7},\text{C1}} + 0.3744642223\,x_{\text{S7},\text{C2}} + 2079.466865\,x_{\text{S7},\text{C3}} + 0.3065671756\,x_{\text{S7},\text{C4}} + 1031.777296\,x_{\text{S7},\text{C5}} \\
&+ 7.203964492\,x_{\text{S7},\text{C6}} + 0.0762307224\,x_{\text{S7},\text{C7}} + 0.0324738795\,x_{\text{S7},\text{C8}} + 23.68582797\,x_{\text{S7},\text{C9}} + 849.9799407\,x_{\text{S7},\text{C10}} \\
&+ 56.09931097\,x_{\text{S8},\text{C1}} + 935.6143109\,x_{\text{S8},\text{C2}} + 73.08824617\,x_{\text{S8},\text{C3}} + 52.00392409\,x_{\text{S8},\text{C4}} + 4.025792389\,x_{\text{S8},\text{C5}} \\
&+ 1002.232766\,x_{\text{S8},\text{C6}} + 935.7766030\,x_{\text{S8},\text{C7}} + 935.7007252\,x_{\text{S8},\text{C8}} + 612.8698719\,x_{\text{S8},\text{C9}} + 1348.836615\,x_{\text{S8},\text{C10}} \\
&+ 4.502283327\,x_{\text{S9},\text{C1}} + 0.3899585343\,x_{\text{S9},\text{C2}} + 1782.466218\,x_{\text{S9},\text{C3}} + 0.006345907\,x_{\text{S9},\text{C4}} + 1031.991011\,x_{\text{S9},\text{C5}} \\
&+ 129.5066562\,x_{\text{S9},\text{C6}} + 0.2118319573\,x_{\text{S9},\text{C7}} + 0.6457301074\,x_{\text{S9},\text{C8}} + 497.6272391\,x_{\text{S9},\text{C9}} + 40.46575555\,x_{\text{S9},\text{C10}} \\
&+ 333.6869270\,x_{\text{S10},\text{C1}} + 277.4719386\,x_{\text{S10},\text{C2}} + 86.02096892\,x_{\text{S10},\text{C3}} + 277.3083661\,x_{\text{S10},\text{C4}} + 1004.464908\,x_{\text{S10},\text{C5}} \\
&+ 19.95033682\,x_{\text{S10},\text{C6}} + 13.20207369\,x_{\text{S10},\text{C7}} + 238.1432152\,x_{\text{S10},\text{C8}} + 411.0580332\,x_{\text{S10},\text{C9}} + 941.7526366\,x_{\text{S10},\text{C10}}
\end{aligned}
\Bigg)
\]

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):

For each $j \in J$:
\[
\sum_{i \in I} x_{ij} \geq d_j
\]
That is,
\[
\begin{aligned}
&x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + \cdots + x_{\text{S10},\text{C1}} \geq 216 \\
&x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S10},\text{C2}} \geq 168 \\
&x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + \cdots + x_{\text{S10},\text{C3}} \geq 264 \\
&x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + \cdots + x_{\text{S10},\text{C4}} \geq 216 \\
&x_{\text{S1},\text{C5}} + x_{\text{S2},\text{C5}} + \cdots + x_{\text{S10},\text{C5}} \geq 216 \\
&x_{\text{S1},\text{C6}} + x_{\text{S2},\text{C6}} + \cdots + x_{\text{S10},\text{C6}} \geq 192 \\
&x_{\text{S1},\text{C7}} + x_{\text{S2},\text{C7}} + \cdots + x_{\text{S10},\text{C7}} \geq 144 \\
&x_{\text{S1},\text{C8}} + x_{\text{S2},\text{C8}} + \cdots + x_{\text{S10},\text{C8}} \geq 168 \\
&x_{\text{S1},\text{C9}} + x_{\text{S2},\text{C9}} + \cdots + x_{\text{S10},\text{C9}} \geq 168 \\
&x_{\text{S1},\text{C10}} + x_{\text{S2},\text{C10}} + \cdots + x_{\text{S10},\text{C10}} \geq 168 \\
\end{aligned}
\]

2. **Supply capacity** (each supplier does not exceed its capacity):

For each $i \in I$:
\[
\sum_{j \in J} x_{ij} \leq s_i
\]
That is,
\[
\begin{aligned}
&x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + \cdots + x_{\text{S1},\text{C10}} \leq 288 \\
&x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S2},\text{C10}} \leq 288 \\
&x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + \cdots + x_{\text{S3},\text{C10}} \leq 264 \\
&x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + \cdots + x_{\text{S4},\text{C10}} \leq 264 \\
&x_{\text{S5},\text{C1}} + x_{\text{S5},\text{C2}} + \cdots + x_{\text{S5},\text{C10}} \leq 216 \\
&x_{\text{S6},\text{C1}} + x_{\text{S6},\text{C2}} + \cdots + x_{\text{S6},\text{C10}} \leq 216 \\
&x_{\text{S7},\text{C1}} + x_{\text{S7},\text{C2}} + \cdots + x_{\text{S7},\text{C10}} \leq 168 \\
&x_{\text{S8},\text{C1}} + x_{\text{S8},\text{C2}} + \cdots + x_{\text{S8},\text{C10}} \leq 216 \\
&x_{\text{S9},\text{C1}} + x_{\text{S9},\text{C2}} + \cdots + x_{\text{S9},\text{C10}} \leq 240 \\
&x_{\text{S10},\text{C1}} + x_{\text{S10},\text{C2}} + \cdots + x_{\text{S10},\text{C10}} \leq 168 \\
\end{aligned}
\]

3. **Non-negativity**:

\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

---

**Summary:**  
Minimize total transportation cost subject to meeting all customer demands, not exceeding any supplier's capacity, and non-negativity of all shipment variables, using the exact coefficients and identifiers from the provided data.