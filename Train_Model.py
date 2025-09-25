import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


#----------------------------------------------------------


#Establish the dataset in a pandas dataframe
df = pd.read_csv("synthetic_rear_area_attack_with_route_and_threats.csv")

target_column = "attack_probability"
#Define the dopped columns
drops_cols = ["artillery_attack","uav_risk","ambush_risk","mine_risk","base_risk",target_column]

#Put the column info into the pandas dataframe for training
x_data = df.drop(columns=drops_cols, errors="ignore")
y_data = df[target_column]

#-------------------------------------------------------

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x_data)
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1,1))

x_train,x_test,y_train,y_test = train_test_split(
    x_scaled,y_scaled,train_size=0.8,random_state=2)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


#-------------------------------------------------------

torch.manual_seed(42)

input_size = x_data.shape[1] #automatically detects number of features

class NN_Regression(nn.Module):
    def __init__(self, input_size):
        super(NN_Regression, self).__init__()
        self.layer1 = nn.Linear(input_size,64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,8)
        self.layer5 = nn.Linear(8,4)
        self.layer6 = nn.Linear(4,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = self.layer6(x)
        return x

#----------------------------------------------------------

#----------------------------------------------------------

#Pick Model
model = NN_Regression(input_size)

optimiser = optim.Adam(model.parameters(), lr=0.01) #load optimiser

#-------------------------------------------------------
#predict
predictions = model(x_train)

loss = nn.MSELoss()

if __name__ == "__main__":

    num_epochs = 1000

    for epoch in range(num_epochs):
        predictions = model(x_train)
        MSE = loss(predictions,y_train.view(-1,1))
        MSE.backward()
        optimiser.step()
        optimiser.zero_grad()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] MSE Loss: {MSE.item()}")


    model.eval()

    with torch.no_grad():
        predicted_values = model(x_test)
        eval_MSE = loss(predicted_values,y_test.view(-1,1))
        # inverse transform predictions and actuals to original scale
        predicted_unscaled = scaler_y.inverse_transform(predicted_values.numpy())
        actual_unscaled = scaler_y.inverse_transform(y_test.view(-1,1))

        print("\nSample Predictions Vs actual values:")

    for i in range(10):
        pred = predicted_unscaled[i][0]
        actual = actual_unscaled[i][0]
        print(f"Prediction: {pred: .2f}, Actual: {actual: .2f}")

    print(f"\nFinal MSE: {eval_MSE.item():.2f}")

    r2 = r2_score(actual_unscaled, predicted_unscaled)

    print(f"\nR² Score: {r2: .4f}")

#-----------------------------------------------------------------------

    torch.save(model.state_dict(), "horizon_attack_model.pt")

    with open("scaler_x.pkl", "wb") as f:
        pickle.dump(scaler_x, f)

    with open("scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)


