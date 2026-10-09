##### Sets

- $I = \{\text{SC1}, \text{SC2}, \ldots, \text{SC10}\}$: set of candidate service centres  
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$: set of customers

##### Parameters

- Fixed opening costs for each centre $i \in I$ (from service_centers_fixed_costs.csv):

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

- Service cost for assigning customer $j$ to centre $i$ (from expanded_customer_service_costs.csv):

\[
\begin{align*}
&c_{\text{SC1},j} = [15.1, 13.4, 15.2, 16.8, 13.4, 12.5, 12.1, 12.3, 16.3, 12.1, 16.7, 11.3, 15.1, 8.3, 12.1] \\
&c_{\text{SC2},j} = [21.2, 16.3, 18.8, 19.1, 18.6, 22.5, 17.1, 15.7, 21.3, 18.7, 18.7, 23.8, 20.5, 20.7, 16.3] \\
&c_{\text{SC3},j} = [14.9, 20.2, 14.7, 18.3, 20.8, 15.5, 19.8, 17.9, 17.6, 14.4, 15.7, 15.5, 15.1, 14.7, 16.4] \\
&c_{\text{SC4},j} = [18.8, 19.6, 21.7, 18.8, 19.8, 14.9, 18.6, 21.3, 20.8, 20.1, 19.9, 17.3, 18.4, 20.4, 15.1] \\
&c_{\text{SC5},j} = [22.9, 20.9, 18.1, 23.1, 22.1, 21.6, 22.1, 22.7, 21.8, 22.7, 24.2, 23.2, 20.6, 20.6, 21.3] \\
&c_{\text{SC6},j} = [16.8, 22.1, 18.6, 15.7, 18.1, 21.3, 20.7, 15.3, 17.2, 14.1, 18.7, 17.7, 17.9, 14.8, 19.1] \\
&c_{\text{SC7},j} = [16.5, 16.9, 12.3, 13.1, 16.7, 16.1, 20.5, 16.6, 15.5, 18.1, 14.2, 16.8, 14.5, 14.2, 19.5] \\
&c_{\text{SC8},j} = [9.4, 9.4, 11.2, 8.6, 12.1, 10.7, 12.2, 11.4, 12.6, 11.4, 13.1, 14.5, 8.5, 11.5, 16.7] \\
&c_{\text{SC9},j} = [16.1, 13.8, 11.9, 15.6, 11.4, 11.9, 15.4, 14.1, 19.9, 18.1, 14.7, 15.8, 14.9, 14.1, 11.1] \\
&c_{\text{SC10},j} = [17.3, 11.7, 20.4, 22.2, 18.2, 14.6, 18.7, 20.1, 19.1, 17.4, 16.1, 17.8, 13.9, 15.1, 18.7] \\
\end{align*}
\]

where the $j$-th entry in each vector corresponds to customer $Cj$ for $j=1,\ldots,15$.

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if customer $j$ is assigned to centre $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Each customer is assigned to exactly one centre:**
   \[
   \sum_{i \in I} x_{ij} = 1 \qquad \forall j \in J
   \]

2. **Customers can only be assigned to open centres:**
   \[
   x_{ij} \leq y_i \qquad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i \qquad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

##### Parameters (full data):

- $I = \{\text{SC1}, \text{SC2}, \text{SC3}, \text{SC4}, \text{SC5}, \text{SC6}, \text{SC7}, \text{SC8}, \text{SC9}, \text{SC10}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}, \text{C13}, \text{C14}, \text{C15}\}$

- $f_i$ and $c_{ij}$ as listed above.

This is the complete mathematical model for the described facility location problem with assignment and capacity constraints.