**Mathematical Model**

Let  

- \(I\) be the set of vehicle types.  
- \(b_i\) be the benefit coefficient for vehicle type \(i\).  
- \(c_i\) be the daily inventory limit (capacity) for vehicle type \(i\).  
- \(X_i\) be the number of units of type \(i\) ordered per day.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} b_i\,X_i \\[4pt]
\text{s.t.}\quad 
& \sum_{i\in I} X_i \;\le\; \sum_{i\in I} c_i \\[4pt]
& X_i \;\le\; c_i \qquad \forall i\in I \\[4pt]
& X_i \in \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "tables": [
    {
      "table_id": "file_0_view_0",
      "columns": {
        "VehicleID": "id",
        "VehicleType": "type",
        "Capacity": "c_i"
      },
      "index": "VehicleType",
      "parameter": "c_i"
    },
    {
      "table_id": "file_1_view_0",
      "columns": {
        "ProductName": "type",
        "Value": "b_i",
        "Weight": "w_i"
      },
      "index": "ProductName",
      "parameter": "b_i"
    }
  ],
  "index_set": {
    "I": "VehicleType / ProductName"
  }
}
```

*The `VehicleType` from `capacity.csv` and `ProductName` from `products.csv` are matched by name to form the index set \(I\). Parameters `c_i` and `b_i` are taken directly from the corresponding columns.*