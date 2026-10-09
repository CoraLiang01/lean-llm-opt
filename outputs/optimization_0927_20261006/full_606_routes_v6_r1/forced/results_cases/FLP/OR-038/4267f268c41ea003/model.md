##### Sets and Indices

Let $I$ be the set of vehicle types:
$$
I = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}
$$

##### Parameters

For each $i \in I$:

- $v_i$: benefit coefficient (from products.csv)
- $u_i$: daily inventory limit (from capacity.csv)

The parameters are:

| Vehicle Type         | $v_i$ (Benefit) | $u_i$ (Capacity) |
|---------------------|-----------------|------------------|
| Sedans              | 1200            | 100              |
| SUVs                | 1800            | 80               |
| Electric Vehicles   | 2500            | 120              |
| Hybrid Vehicles     | 2000            | 90               |
| Trucks              | 1500            | 50               |
| Sports Cars         | 3000            | 30               |
| Compact Cars        | 1000            | 110              |
| Luxury Sedans       | 3500            | 40               |
| Vans                | 1600            | 60               |
| Pickup Trucks       | 1700            | 35               |

Let $C$ denote the total inventory capacity per day. (If not specified in the data, this should be provided externally.)

##### Decision Variables

For each $i \in I$:

- $x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to order per day

##### Mathematical Model

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
$$

**Subject to:**

1. **Individual vehicle type limits:**
   $$
   0 \leq x_i \leq u_i, \quad \forall i \in I
   $$
2. **Total inventory capacity:**
   $$
   \sum_{i \in I} x_i \leq C
   $$
3. **Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

##### Parameters (explicit values):

- $v_{\text{Sedans}} = 1200$, $u_{\text{Sedans}} = 100$
- $v_{\text{SUVs}} = 1800$, $u_{\text{SUVs}} = 80$
- $v_{\text{Electric Vehicles}} = 2500$, $u_{\text{Electric Vehicles}} = 120$
- $v_{\text{Hybrid Vehicles}} = 2000$, $u_{\text{Hybrid Vehicles}} = 90$
- $v_{\text{Trucks}} = 1500$, $u_{\text{Trucks}} = 50$
- $v_{\text{Sports Cars}} = 3000$, $u_{\text{Sports Cars}} = 30$
- $v_{\text{Compact Cars}} = 1000$, $u_{\text{Compact Cars}} = 110$
- $v_{\text{Luxury Sedans}} = 3500$, $u_{\text{Luxury Sedans}} = 40$
- $v_{\text{Vans}} = 1600$, $u_{\text{Vans}} = 60$
- $v_{\text{Pickup Trucks}} = 1700$, $u_{\text{Pickup Trucks}} = 35$

##### Notes

- The total inventory capacity $C$ must be specified for the model to be complete.
- All variables $x_i$ are integer and nonnegative.

###### Retrieved Information

{
  "vehicle_types": [
    "Sedans",
    "SUVs",
    "Electric Vehicles",
    "Hybrid Vehicles",
    "Trucks",
    "Sports Cars",
    "Compact Cars",
    "Luxury Sedans",
    "Vans",
    "Pickup Trucks"
  ],
  "benefit": {
    "Sedans": 1200,
    "SUVs": 1800,
    "Electric Vehicles": 2500,
    "Hybrid Vehicles": 2000,
    "Trucks": 1500,
    "Sports Cars": 3000,
    "Compact Cars": 1000,
    "Luxury Sedans": 3500,
    "Vans": 1600,
    "Pickup Trucks": 1700
  },
  "capacity": {
    "Sedans": 100,
    "SUVs": 80,
    "Electric Vehicles": 120,
    "Hybrid Vehicles": 90,
    "Trucks": 50,
    "Sports Cars": 30,
    "Compact Cars": 110,
    "Luxury Sedans": 40,
    "Vans": 60,
    "Pickup Trucks": 35
  }
}