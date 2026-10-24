##### Objective Function:

$\quad \min \sum_{(w,p) \in E} c_{w,p} \cdot x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $E$ is the set of eligible (worker, project) pairs as defined below.

##### Constraints

###### 1. Project Coverage

$\sum_{w: (w,p) \in E} x_{w,p} = 1 \quad \forall p \in P$

(Each project is assigned exactly one eligible worker.)

###### 2. Worker Assignment

$\sum_{p: (w,p) \in E} x_{w,p} \leq 1 \quad \forall w \in W$

(Each worker is assigned to at most one project.)

###### 3. Eligibility Mask

$x_{w,p} = 0$ for all $(w,p) \notin E$

(Only listed offers, with eligible workers and sufficient skill, are permitted.)

###### 4. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in E$

##### Eligibility Mask $E$

A pair $(w,p)$ is in $E$ if and only if:

- The offer $(w,p)$ is listed in the data (i.e., there is a cost for this pair).
- Worker $w$ is not on leave ($\text{on_leave}_w = 0$).
- Worker $w$'s skill is at least as high as the required skill for project $p$ (using the order: Junior < Intermediate < Senior < Expert).

##### Retrieved Information

###### Workers

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W01       | Senior      | 1        |
| W00       | Junior      | 0        |
| W19       | Junior      | 0        |
| W11       | Senior      | 0        |
| W04       | Junior      | 0        |
| W15       | Expert      | 0        |
| W10       | Expert      | 0        |
| W06       | Expert      | 0        |
| W20       | Intermediate| 0        |
| W14       | Intermediate| 0        |

###### Projects

| project_id | required_skill |
|------------|---------------|
| P00        | Intermediate   |
| P01        | Intermediate   |
| P02        | Senior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Junior        |
| P06        | Junior        |
| P07        | Junior        |

###### Skill Order

Junior = 1, Intermediate = 2, Senior = 3, Expert = 4

###### Offer List (Eligible Pairs Only)

Below, only pairs where the worker is not on leave and has sufficient skill are included. (Pairs with W01 are excluded due to on_leave=1.)

- W00 (Junior, on_leave=0): Can only do projects with required_skill=Junior (P03, P04, P05, P06, P07)
    - Offers:
        - (W00, P01): 17 (required_skill=Intermediate) → Not eligible (Junior < Intermediate)
        - (W00, P04): 430 (Junior) → Eligible
        - (W00, P05): 1385 (Junior) → Eligible
        - (W00, P06): 260 (Junior) → Eligible
        - (W00, P07): 1107 (Junior) → Eligible

- W19 (Junior, on_leave=0): Can only do projects with required_skill=Junior (P03, P04, P05, P06, P07)
    - Offers:
        - (W19, P00): 5 (Intermediate) → Not eligible
        - (W19, P02): 9 (Senior) → Not eligible
        - (W19, P03): 109 (Junior) → Eligible
        - (W19, P04): 1306 (Junior) → Eligible
        - (W19, P05): 626 (Junior) → Eligible
        - (W19, P06): 466 (Junior) → Eligible
        - (W19, P07): 1365 (Junior) → Eligible

- W11 (Senior, on_leave=0): Can do projects with required_skill ≤ Senior (all projects)
    - Offers:
        - (W11, P00): 235 (Intermediate) → Eligible
        - (W11, P02): 1034 (Senior) → Eligible
        - (W11, P03): 782 (Junior) → Eligible
        - (W11, P04): 556 (Junior) → Eligible
        - (W11, P05): 1018 (Junior) → Eligible
        - (W11, P07): 1136 (Junior) → Eligible

- W04 (Junior, on_leave=0): Can only do projects with required_skill=Junior (P03, P04, P05, P06, P07)
    - Offers:
        - (W04, P00): 4 (Intermediate) → Not eligible
        - (W04, P01): 8 (Intermediate) → Not eligible
        - (W04, P02): 16 (Senior) → Not eligible
        - (W04, P03): 543 (Junior) → Eligible
        - (W04, P04): 205 (Junior) → Eligible
        - (W04, P05): 1149 (Junior) → Eligible
        - (W04, P06): 533 (Junior) → Eligible
        - (W04, P07): 732 (Junior) → Eligible

- W15 (Expert, on_leave=0): Can do all projects
    - Offers:
        - (W15, P00): 722 (Intermediate) → Eligible
        - (W15, P02): 577 (Senior) → Eligible
        - (W15, P03): 1062 (Junior) → Eligible
        - (W15, P04): 1314 (Junior) → Eligible
        - (W15, P05): 528 (Junior) → Eligible
        - (W15, P06): 835 (Junior) → Eligible
        - (W15, P07): 918 (Junior) → Eligible

- W10 (Expert, on_leave=0): Can do all projects
    - Offers:
        - (W10, P00): 147 (Intermediate) → Eligible
        - (W10, P01): 114 (Intermediate) → Eligible
        - (W10, P03): 998 (Junior) → Eligible
        - (W10, P04): 1270 (Junior) → Eligible
        - (W10, P07): 953 (Junior) → Eligible

- W06 (Expert, on_leave=0): Can do all projects
    - Offers:
        - (W06, P00): 1217 (Intermediate) → Eligible
        - (W06, P01): 425 (Intermediate) → Eligible
        - (W06, P02): 1320 (Senior) → Eligible
        - (W06, P03): 1097 (Junior) → Eligible
        - (W06, P04): 822 (Junior) → Eligible
        - (W06, P05): 221 (Junior) → Eligible
        - (W06, P06): 496 (Junior) → Eligible
        - (W06, P07): 908 (Junior) → Eligible

- W20 (Intermediate, on_leave=0): Can do projects with required_skill ≤ Intermediate (P00, P01, P03, P04, P05, P06, P07)
    - Offers:
        - (W20, P00): 675 (Intermediate) → Eligible
        - (W20, P01): 1055 (Intermediate) → Eligible
        - (W20, P02): 8 (Senior) → Not eligible
        - (W20, P03): 703 (Junior) → Eligible
        - (W20, P06): 893 (Junior) → Eligible
        - (W20, P07): 129 (Junior) → Eligible

- W14 (Intermediate, on_leave=0): Can do projects with required_skill ≤ Intermediate (P00, P01, P03, P04, P05, P06, P07)
    - Offers:
        - (W14, P01): 1361 (Intermediate) → Eligible
        - (W14, P03): 379 (Junior) → Eligible
        - (W14, P04): 239 (Junior) → Eligible
        - (W14, P05): 460 (Junior) → Eligible
        - (W14, P06): 130 (Junior) → Eligible

###### Complete Eligible Offer List $E$ and Costs $c_{w,p}$

| worker_id | project_id | cost_cents |
|-----------|------------|------------|
| W00       | P04        | 430        |
| W00       | P05        | 1385       |
| W00       | P06        | 260        |
| W00       | P07        | 1107       |
| W19       | P03        | 109        |
| W19       | P04        | 1306       |
| W19       | P05        | 626        |
| W19       | P06        | 466        |
| W19       | P07        | 1365       |
| W11       | P00        | 235        |
| W11       | P02        | 1034       |
| W11       | P03        | 782        |
| W11       | P04        | 556        |
| W11       | P05        | 1018       |
| W11       | P07        | 1136       |
| W04       | P03        | 543        |
| W04       | P04        | 205        |
| W04       | P05        | 1149       |
| W04       | P06        | 533        |
| W04       | P07        | 732        |
| W15       | P00        | 722        |
| W15       | P02        | 577        |
| W15       | P03        | 1062       |
| W15       | P04        | 1314       |
| W15       | P05        | 528        |
| W15       | P06        | 835        |
| W15       | P07        | 918        |
| W10       | P00        | 147        |
| W10       | P01        | 114        |
| W10       | P03        | 998        |
| W10       | P04        | 1270       |
| W10       | P07        | 953        |
| W06       | P00        | 1217       |
| W06       | P01        | 425        |
| W06       | P02        | 1320       |
| W06       | P03        | 1097       |
| W06       | P04        | 822        |
| W06       | P05        | 221        |
| W06       | P06        | 496        |
| W06       | P07        | 908        |
| W20       | P00        | 675        |
| W20       | P01        | 1055       |
| W20       | P03        | 703        |
| W20       | P06        | 893        |
| W20       | P07        | 129        |
| W14       | P01        | 1361       |
| W14       | P03        | 379        |
| W14       | P04        | 239        |
| W14       | P05        | 460        |
| W14       | P06        | 130        |

##### Sets

- $W$ = {W00, W19, W11, W04, W15, W10, W06, W20, W14}
- $P$ = {P00, P01, P02, P03, P04, P05, P06, P07}
- $E$ = set of eligible pairs as above

##### Variables

- $x_{w,p} \in \{0,1\}$ for each $(w,p) \in E$

##### Parameters

- $c_{w,p}$: cost in cents for each $(w,p) \in E$ (see table above)

##### Model Summary

$\boxed{
\begin{align*}
\min \quad & \sum_{(w,p) \in E} c_{w,p} \cdot x_{w,p} \\
\text{s.t.} \quad & \sum_{w: (w,p) \in E} x_{w,p} = 1 \quad \forall p \in P \\
& \sum_{p: (w,p) \in E} x_{w,p} \leq 1 \quad \forall w \in W \\
& x_{w,p} = 0 \quad \forall (w,p) \notin E \\
& x_{w,p} \in \{0,1\} \quad \forall (w,p) \in E \\
\end{align*}
}$

All costs are in USD cents. The minimum value of the objective is the minimum total assignment cost in cents.