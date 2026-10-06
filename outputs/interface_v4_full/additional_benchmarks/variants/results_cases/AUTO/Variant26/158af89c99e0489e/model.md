##### Decision Variables

Let $x_{w,p}$ be a binary variable, where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, and $0$ otherwise. Only pairs $(w,p)$ listed in the eligible offer table below are permitted.

##### Objective Function

$\min \sum_{(w,p) \in \mathcal{E}} \text{cost}_{w,p} \cdot x_{w,p}$

where $\mathcal{E}$ is the set of all eligible (worker, project) pairs as listed below, and $\text{cost}_{w,p}$ is the cost in USD cents for worker $w$ to perform project $p$.

##### Constraints

1. **Each project is assigned to exactly one worker:**

$\sum_{w: (w,p) \in \mathcal{E}} x_{w,p} = 1 \quad \forall p \in \{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}, \text{P06}, \text{P07}\}$

2. **Each worker is assigned to at most one project:**

$\sum_{p: (w,p) \in \mathcal{E}} x_{w,p} \leq 1 \quad \forall w \in \{\text{W00}, \text{W04}, \text{W06}, \text{W10}, \text{W11}, \text{W14}, \text{W15}, \text{W19}, \text{W20}\}$

3. **Assignment only allowed for eligible offers:**

$x_{w,p} = 0$ for all $(w,p) \notin \mathcal{E}$

4. **Variable domain:**

$x_{w,p} \in \{0,1\}$ for all $(w,p) \in \mathcal{E}$

##### Retrieved Information

**Eligible worker-project pairs and costs (in USD cents):**

| worker_id | worker_skill  | project_id | required_skill | cost_cents |
|-----------|--------------|------------|---------------|------------|
| W06       | Expert       | P00        | Intermediate  | 1217       |
| W10       | Expert       | P00        | Intermediate  | 147        |
| W11       | Senior       | P00        | Intermediate  | 235        |
| W15       | Expert       | P00        | Intermediate  | 722        |
| W20       | Intermediate | P00        | Intermediate  | 675        |
| W06       | Expert       | P01        | Intermediate  | 425        |
| W10       | Expert       | P01        | Intermediate  | 114        |
| W14       | Intermediate | P01        | Intermediate  | 1361       |
| W20       | Intermediate | P01        | Intermediate  | 1055       |
| W06       | Expert       | P02        | Senior        | 1320       |
| W11       | Senior       | P02        | Senior        | 1034       |
| W15       | Expert       | P02        | Senior        | 577        |
| W04       | Junior       | P03        | Junior        | 543        |
| W06       | Expert       | P03        | Junior        | 1097       |
| W10       | Expert       | P03        | Junior        | 998        |
| W11       | Senior       | P03        | Junior        | 782        |
| W14       | Intermediate | P03        | Junior        | 379        |
| W15       | Expert       | P03        | Junior        | 1062       |
| W19       | Junior       | P03        | Junior        | 109        |
| W20       | Intermediate | P03        | Junior        | 703        |
| W00       | Junior       | P04        | Junior        | 430        |
| W04       | Junior       | P04        | Junior        | 205        |
| W06       | Expert       | P04        | Junior        | 822        |
| W10       | Expert       | P04        | Junior        | 1270       |
| W11       | Senior       | P04        | Junior        | 556        |
| W14       | Intermediate | P04        | Junior        | 239        |
| W15       | Expert       | P04        | Junior        | 1314       |
| W19       | Junior       | P04        | Junior        | 1306       |
| W00       | Junior       | P05        | Junior        | 1385       |
| W04       | Junior       | P05        | Junior        | 1149       |
| W06       | Expert       | P05        | Junior        | 221        |
| W11       | Senior       | P05        | Junior        | 1018       |
| W14       | Intermediate | P05        | Junior        | 460        |
| W15       | Expert       | P05        | Junior        | 528        |
| W19       | Junior       | P05        | Junior        | 626        |
| W00       | Junior       | P06        | Junior        | 260        |
| W04       | Junior       | P06        | Junior        | 533        |
| W06       | Expert       | P06        | Junior        | 496        |
| W14       | Intermediate | P06        | Junior        | 130        |
| W15       | Expert       | P06        | Junior        | 835        |
| W19       | Junior       | P06        | Junior        | 466        |
| W20       | Intermediate | P06        | Junior        | 893        |
| W00       | Junior       | P07        | Junior        | 1107       |
| W04       | Junior       | P07        | Junior        | 732        |
| W06       | Expert       | P07        | Junior        | 908        |
| W10       | Expert       | P07        | Junior        | 953        |
| W11       | Senior       | P07        | Junior        | 1136       |
| W15       | Expert       | P07        | Junior        | 918        |
| W19       | Junior       | P07        | Junior        | 1365       |
| W20       | Intermediate | P07        | Junior        | 129        |

**Sets:**

- Workers: $\{\text{W00}, \text{W04}, \text{W06}, \text{W10}, \text{W11}, \text{W14}, \text{W15}, \text{W19}, \text{W20}\}$
- Projects: $\{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}, \text{P06}, \text{P07}\}$
- Eligible assignments: $\mathcal{E}$ as listed above.

**Parameters:**

- $\text{cost}_{w,p}$: as in the table above, in USD cents.

##### Model Summary

$\boxed{
\begin{align*}
\min \quad & \sum_{(w,p) \in \mathcal{E}} \text{cost}_{w,p} \cdot x_{w,p} \\
\text{s.t.} \quad & \sum_{w: (w,p) \in \mathcal{E}} x_{w,p} = 1 \quad \forall p \\
& \sum_{p: (w,p) \in \mathcal{E}} x_{w,p} \leq 1 \quad \forall w \\
& x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{E} \\
\end{align*}
}$

All costs are in USD cents. Only assignments listed in the eligible offer table are permitted. Each project is assigned to exactly one worker, and each worker to at most one project.