{
  "query": "A water authority must install monitoring sensors so every monitoring zone is covered. Candidate sensor costs and covered zones are listed in sensor_sites.csv, and the complete set of monitoring zones is listed in monitoring_zones.csv.\n\nFormulate a minimum-cost set covering model. For each candidate sensor site i, define y_i as a binary variable equal to 1 if a sensor is installed at site i. The objective is to minimize total installation cost. The model should include one coverage constraint for each monitoring zone requiring at least one selected site that covers that zone, and binary restrictions for all sensor-site variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Center",
        "OpeningCost",
        "CoveredDistricts"
      ],
      "file_name": "sensor_sites.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Center": "S1",
            "CoveredDistricts": "Q1;Q2;Q4",
            "OpeningCost": "8"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Center": "S2",
            "CoveredDistricts": "Q2;Q3;Q5",
            "OpeningCost": "12"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Center": "S3",
            "CoveredDistricts": "Q4;Q6;Q7",
            "OpeningCost": "11"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Center": "S4",
            "CoveredDistricts": "Q5;Q8",
            "OpeningCost": "10"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Center": "S5",
            "CoveredDistricts": "Q6;Q8;Q9",
            "OpeningCost": "13"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Center": "S6",
            "CoveredDistricts": "Q7;Q10;Q11",
            "OpeningCost": "9"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Center": "S7",
            "CoveredDistricts": "Q1;Q9;Q10",
            "OpeningCost": "14"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Center": "S8",
            "CoveredDistricts": "Q3;Q5;Q11",
            "OpeningCost": "7"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Center": "S9",
            "CoveredDistricts": "Q2;Q6;Q10",
            "OpeningCost": "15"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Zone"
      ],
      "file_name": "monitoring_zones.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Zone": "Q1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Zone": "Q2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Zone": "Q3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Zone": "Q4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Zone": "Q5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Zone": "Q6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Zone": "Q7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Zone": "Q8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Zone": "Q9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Zone": "Q10"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Zone": "Q11"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
