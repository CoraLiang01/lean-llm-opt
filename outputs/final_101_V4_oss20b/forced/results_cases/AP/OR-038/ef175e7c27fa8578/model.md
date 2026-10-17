**Mathematical Model**

Let  

- \(I\) be the set of vehicle types (identified by the *VehicleType* field).  
- \(b_i\) be the benefit coefficient for type \(i\) (from *products.csv*).  
- \(c_i\) be the daily inventory limit for type \(i\) (from *capacity.csv*).  
- \(x_i\) be the number of units of type \(i\) ordered per day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; c_i && \forall i\in I \\[4pt]
             & \sum_{i\in I} x_i \;\le\; \sum_{i\in I} c_i && \\[4pt]
             & x_i \;\in\; \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "parameters": {
    "b": {
      "source": "file_1_view_0",
      "key_column": "ProductName",
      "value_column": "Value"
    },
    "c": {
      "source": "file_0_view_0",
      "key_column": "VehicleType",
      "value_column": "Capacity"
    }
  },
  "index_set": {
    "I": {
      "source": ["file_0_view_0", "file_1_view_0"],
      "key_column": "VehicleType",
      "unique": true
    }
  }
}
```

*The mapping links each vehicle type to its benefit coefficient and daily capacity, ensuring the model uses the exact data from the provided CSV files.*