{
  "query": "A manufacturer plans production of a seasonal spare part over 12 months. Monthly demand, unit production cost, setup cost, holding cost, and production capacity are listed in monthly_lot_sizing.csv. Initial inventory is zero, backlogging is not allowed, and ending inventory after the last month must be zero.\n\nFormulate a minimum-cost capacitated lot-sizing model. For each month t, define x_t as the nonnegative production quantity, inv_t as the nonnegative ending inventory, and y_t as a binary setup variable equal to 1 if production occurs in month t. The objective is to minimize total production, setup, and holding cost. The model should include monthly inventory-balance constraints, production-capacity-to-setup linking constraints, the zero-ending-inventory constraint, nonnegativity constraints for production and inventory variables, and binary restrictions for setup variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Month",
        "Demand",
        "ProductionCost",
        "SetupCost",
        "HoldingCost",
        "ProductionCapacity"
      ],
      "file_name": "monthly_lot_sizing.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Demand": "95",
            "HoldingCost": "1.0",
            "Month": "M01",
            "ProductionCapacity": "310",
            "ProductionCost": "17",
            "SetupCost": "760"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Demand": "70",
            "HoldingCost": "1.1",
            "Month": "M02",
            "ProductionCapacity": "260",
            "ProductionCost": "18",
            "SetupCost": "820"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Demand": "130",
            "HoldingCost": "1.2",
            "Month": "M03",
            "ProductionCapacity": "320",
            "ProductionCost": "20",
            "SetupCost": "930"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Demand": "85",
            "HoldingCost": "1.0",
            "Month": "M04",
            "ProductionCapacity": "300",
            "ProductionCost": "19",
            "SetupCost": "780"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Demand": "115",
            "HoldingCost": "1.3",
            "Month": "M05",
            "ProductionCapacity": "340",
            "ProductionCost": "21",
            "SetupCost": "960"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Demand": "140",
            "HoldingCost": "1.4",
            "Month": "M06",
            "ProductionCapacity": "360",
            "ProductionCost": "22",
            "SetupCost": "1040"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Demand": "90",
            "HoldingCost": "1.1",
            "Month": "M07",
            "ProductionCapacity": "290",
            "ProductionCost": "18",
            "SetupCost": "800"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Demand": "125",
            "HoldingCost": "1.2",
            "Month": "M08",
            "ProductionCapacity": "330",
            "ProductionCost": "20",
            "SetupCost": "900"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Demand": "75",
            "HoldingCost": "1.0",
            "Month": "M09",
            "ProductionCapacity": "280",
            "ProductionCost": "17",
            "SetupCost": "740"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Demand": "150",
            "HoldingCost": "1.5",
            "Month": "M10",
            "ProductionCapacity": "370",
            "ProductionCost": "23",
            "SetupCost": "1100"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Demand": "105",
            "HoldingCost": "1.1",
            "Month": "M11",
            "ProductionCapacity": "310",
            "ProductionCost": "19",
            "SetupCost": "850"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Demand": "120",
            "HoldingCost": "0.0",
            "Month": "M12",
            "ProductionCapacity": "350",
            "ProductionCost": "21",
            "SetupCost": "980"
          }
        }
      ],
      "table_id": "file_0_view_0"
    }
  ]
}
