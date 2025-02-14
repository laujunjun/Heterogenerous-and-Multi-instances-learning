
class HAN(torch.nn.Module):
    def __init__(self,  input_size,hidden_channels,  heads,pooling_ratio):
        super(HAN, self).__init__()
        self.lin_t=torch.nn.Linear(input_size, hidden_channels[0])
        self.lin_s=torch.nn.Linear(input_size, hidden_channels[0])
        self.lin_i=torch.nn.Linear(input_size, hidden_channels[0])
        self.conv1 = HANConv(-1, hidden_channels[0], metadata=(['T', 'S','I'], [("T", "T-T", "T"),("S", "S-S", "S"),("I", "I-I", "I"),("T", "T-S", "S"),("T", "T-I", "I"),("S", "S-I", "I"),("S", "rev_T-S", "T"),("I", "rev_T-I", "T"),("I", "rev_S-I", "S")]), heads=heads)
        self.pool1 = TopKPooling(in_channels=hidden_channels[0], ratio = pooling_ratio)

        self.conv2 = HANConv(-1, hidden_channels[0], metadata=(['T', 'S','I'], [("T", "T-T", "T"),("S", "S-S", "S"),("I", "I-I", "I"),("T", "T-S", "S"),("T", "T-I", "I"),("S", "S-I", "I"),("S", "rev_T-S", "T"),("I", "rev_T-I", "T"),("I", "rev_S-I", "S")]), heads=heads)
        self.pool2 = TopKPooling(in_channels=hidden_channels[0], ratio = pooling_ratio)

        self.conv3 = HANConv(-1, hidden_channels[0], metadata=(['T', 'S','I'], [("T", "T-T", "T"),("S", "S-S", "S"),("I", "I-I", "I"),("T", "T-S", "S"),("T", "T-I", "I"),("S", "S-I", "I"),("S", "rev_T-S", "T"),("I", "rev_T-I", "T"),("I", "rev_S-I", "S")]), heads=heads)
        self.pool3 = TopKPooling(in_channels=hidden_channels[0], ratio = pooling_ratio)

        self.lin1=torch.nn.Linear(hidden_channels[0]*3, 64)
        self.batchNorm = torch.nn.BatchNorm1d(64)

    def forward(self, data: HeteroData):
        x_dict = {
            "T": self.lin_t(data["T"].feature),
            "S": self.lin_s(data["S"].feature),
            "I": self.lin_i(data["I"].feature),
        } 

        x = self.conv1(x_dict, data.edge_index_dict)
        x = {key: xitem.relu() for key, xitem in x.items()}
        x = self.conv2(x, data.edge_index_dict)
        x = {key: xitem.relu() for key, xitem in x.items()}
        x = {key: gap(xitem,data[key].batch) for key, xitem in x.items()}
        x = torch.cat([x["T"], x["S"], x["I"]], dim=-1)
        x = self.lin1(x).relu()
        x = self.batchNorm(x)
        return x
