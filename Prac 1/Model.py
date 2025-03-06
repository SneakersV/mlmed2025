import torch
from torch.utils.data import Dataset, DataLoader

class MIT_BIH_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :-1].values
        label = int(self.data.iloc[idx, -1])

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.uint8)

class ClassfierModel(torch.nn.Module):
    def __init__(self, n_class = 5):
        super(ClassfierModel, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)

        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.drop1 = torch.nn.Dropout(p=0.2)

        self.conv3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv4 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv5 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.drop2 = torch.nn.Dropout(p=0.2)

        self.activation = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(12*208, 32)
        self.fc2 = torch.nn.Linear(32, n_class)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    
def train_model(model, train_df, batch_size=32, lr=0.001, epochs=5):
    train_loader = DataLoader(MIT_BIH_Dataset(train_df), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = model(inputs.unsqueeze(1).to(device))
            
            loss = criterion(outputs.to(device), labels.to(device))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            losses.append(loss)
            
        print(f"Epoch {epoch+1}/{epochs}, loss: {loss.item()}")
            
    return model, losses

def evaluate(model, test_df):
    test_loader = DataLoader(MIT_BIH_Dataset(test_df), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = model(inputs.unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    return 100 * correct / total