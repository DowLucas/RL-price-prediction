import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda:0")


X = np.load("X.npy")
Y = np.load("y.npy")

#print(X[0])

max_val = X.max()

X = X / max_val
Y = Y / max_val

#print(max_val)
#print(X[0]*max_val)

X = torch.from_numpy(X).view(-1, 10, 6).to(device)
Y = torch.from_numpy(Y).view(-1, 1, 6).to(device)

X = X[:-100]
X_train = X[-100:]

Y = Y[:-100]
Y_train = Y[-100:]

BATCH_SIZE = 16
SEQ_LEN = 10


class LSTMNetwork(nn.Module):
    def __init__(self, ):
        super(LSTMNetwork, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=6,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.lin1 = nn.Linear(64, 128)
        self.lin2 = nn.Linear(128, 6)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        x = self.lin1(r_out[:, -1, :])
        x = self.lin2(x)

        return x


LSTM = LSTMNetwork().to(device)


lossF = nn.MSELoss()
optim = optim.SGD(LSTM.parameters(), lr=1e-3)

EPOCHS = 40

f = open("loss.csv", "w")
f.write("")
f.close()
f = open("loss.csv", "a")
f.write("loss\n")

for _ in range(EPOCHS):
    for i in tqdm(range(0, len(X), BATCH_SIZE)):
        optim.zero_grad()
        batch_x = X[i:i+BATCH_SIZE]
        batch_y = Y[i:i+BATCH_SIZE]
        #print(y)
        try:
            outputs = LSTM(batch_x)
        except:
            pass
        loss = lossF(outputs, batch_y)
        loss.backward()
        optim.step()
    f.write(f"{loss}\n")

f.close()

predictions = LSTM(X_train)

with torch.no_grad():

    X_train = X_train.cpu()
    Y_train = Y_train.cpu()
    predictions = predictions.cpu()

    for n, pred in enumerate(predictions):

        pred = pred.numpy() * max_val
        actual = Y_train[n].numpy() * max_val

        print(f"Prediction: {np.round(pred)}\n"
              f"Actual: {np.round(actual)}")






