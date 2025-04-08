import pandas as pd
import pulp


######################   Data, Vectors and Matrices   #########################
data = pd.read_excel('Data.xlsx', sheet_name=None)

locationDict = data['Nodes'].groupby('LocationTYpe')['Location'].apply(list).to_dict()

Beneficiary_camps = locationDict.get('Beneficiary Camp', [])
Ports = locationDict.get('Port', [])
Suppliers = locationDict.get('Supplier', [])
Warehouses = locationDict.get('Warehouse', [])
commodities = data['Commodities'].iloc[0:, 0].tolist()
nutrients = data['Nutrients'].iloc[0:, 0].tolist()

cij = data['Procurement'].groupby(['Commodity', 'Supplier'])['Procurement price ($/ton)'].first().unstack(fill_value=0).to_dict()

print(cij)
#Cost Matrices:

#aquisition + sea transport cost
acquisition_cost = {
    (row['Commodity'], row['Supplier']): row['Procurement price ($/ton)']
    for _, row in data['Procurement'].iterrows()
}
aq_sea_cost = {}
for _, row in data['SeaTransport'].iterrows():
    i = row['Commodity']
    j = row['Origin']
    p = row['Destination']
    sea_cost = row['SeaTransport cost ($/ton)']

    if (i, j) in acquisition_cost:
        total_cost = acquisition_cost[(i, j)] + sea_cost
        aq_sea_cost[(i, j, p)] = total_cost


#transportation cost from Port p to Warehouse w
land_pw_cost = {
    (row['Origin'], row['Destination']): row['Landtransport cost ($/ton)']
    for _, row in data['LandTransport'].iloc[:11].iterrows()
}

#transportation cost from Warehouse w to Village v
land_wv_cost = {
    (row['Origin'], row['Destination']): row['Landtransport cost ($/ton)']
    for _, row in data['LandTransport'].iloc[11:].iterrows()
}

#procurement capacity
procurement_capacity = {
    (row['Commodity'], row['Supplier']): row['Procurement capacity (ton/month)']
    for _, row in data['Procurement'].iterrows()
}

#nutritional values for each commodity
nutritional_values = {
    (row['Commodity (100g)'], nutrient): row[nutrient]
    for _, row in data['Nutritional values'].iterrows()
    for nutrient in data['Nutritional values'].columns[1:]
}

######################   MODEL   #########################
model = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)

# Decision Variables
x = pulp.LpVariable.dicts("x", aq_sea_cost.keys(), lowBound=0, cat='Continuous')
y = pulp.LpVariable.dicts("y", land_pw_cost.keys(), lowBound=0, cat='Continuous')
z = pulp.LpVariable.dicts("z", land_wv_cost.keys(), lowBound=0, cat='Continuous')


### Cost function ###
model += (
    pulp.lpSum(x[i_j_p] * aq_sea_cost[i_j_p] for i_j_p in aq_sea_cost) +
    pulp.lpSum(y[p_w] * land_pw_cost[p_w] for p_w in land_pw_cost) +
    pulp.lpSum(z[w_v] * land_wv_cost[w_v] for w_v in land_wv_cost)
)


### Constraints ###

#Procurement capacity
for (i, j) in procurement_capacity:
    total_procured = pulp.lpSum(x[i, j, p] for p in Ports if (i, j, p) in x)
    model += total_procured <= procurement_capacity[(i, j)], f"ProcureCap_{i}_{j}"

#Flow conservation
for i in commodities:
    for p in Ports:
        incoming = pulp.lpSum(x[i, j, p] for j in Suppliers if (i, j, p) in x)
        outgoing = pulp.lpSum(y[i, p, w] for w in Warehouses if (i, p, w) in y)
        model += incoming == outgoing, f"Flow_Port_{i}_{p}"

for i in commodities:
    for w in Warehouses:
        incoming = pulp.lpSum(y[i, p, w] for p in Ports if (i, p, w) in y)
        outgoing = pulp.lpSum(z[i, w, v] for v in Beneficiary_camps if (i, w, v) in z)
        model += incoming == outgoing, f"Flow_Warehouse_{i}_{w}"

