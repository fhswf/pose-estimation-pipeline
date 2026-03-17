import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Modelldefinition
class model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation(nn.Module):
    def __init__(self, num_keypoints=133):
        super(model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation, self).__init__()
        self.upscale=nn.Linear(num_keypoints * 2, 1024)
        self.fc1=nn.Linear(1024, 1024)
        self.bn1=nn.BatchNorm1d(1024)
        self.fc2=nn.Linear(1024, 1024)
        self.bn2=nn.BatchNorm1d(1024)
        self.fc3=nn.Linear(1024, 1024)
        self.bn3=nn.BatchNorm1d(1024)
        self.fc4=nn.Linear(1024, 1024)
        self.bn4=nn.BatchNorm1d(1024)
        self.outputlayer=nn.Linear(1024, num_keypoints * 3)

    def forward(self, x):
        x=self.upscale(x)
        x1=nn.Dropout(p=0.5)(nn.ReLU()(self.bn1(self.fc1(x))))
        x1=nn.Dropout(p=0.5)(nn.ReLU()(self.bn2(self.fc2(x1))))
        x=x + x1
        x1=nn.Dropout(p=0.5)(nn.ReLU()(self.bn3(self.fc3(x))))
        x1=nn.Dropout(p=0.5)(nn.ReLU()(self.bn4(self.fc4(x1))))
        x=x + x1
        x=self.outputlayer(x)
        return x


# Dataset Klasse für Ihre Daten - jetzt für Listen optimiert
class KeypointsDataset(Dataset):
    def __init__(self, data_list, num_keypoints=133):
        if num_keypoints not in [133, 17]:
            raise ValueError(f"numkeypoints muss 133 oder 17 sein, nicht {num_keypoints}")

        self.data=[]

        for frame_data in data_list:
            keypoints_2d=frame_data.get('keypoints_2d', {})
            keypoints_3d=frame_data.get('keypoints_3d', {})

            # 2D Keypoints extrahieren
            kp2d_list=[]
            for i in range(num_keypoints):  # Nur die ersten extract_points Punkte
                if str(i) in keypoints_2d:
                    kp2d_list.append(keypoints_2d[str(i)].get('x', 0.0))
                    kp2d_list.append(keypoints_2d[str(i)].get('y', 0.0))
                else:
                    kp2d_list.extend([0.0, 0.0])

            # 3D Keypoints extrahieren
            kp3d_list=[]
            for i in range(num_keypoints):  # Nur die ersten extract_points Punkte
                if str(i) in keypoints_3d:
                    kp3d_list.append(keypoints_3d[str(i)].get('x', 0.0))
                    kp3d_list.append(keypoints_3d[str(i)].get('y', 0.0))
                    kp3d_list.append(keypoints_3d[str(i)].get('z', 0.0))
                else:
                    kp3d_list.extend([0.0, 0.0, 0.0])

            self.data.append({
                'kp2d': np.array(kp2d_list, dtype=np.float32),
                'kp3d': np.array(kp3d_list, dtype=np.float32)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['kp2d'], self.data[idx]['kp3d']


# Evaluierungsfunktion
def evaluate_model(model, data_list, device, num_keypoints=133):
    model.eval()
    dataset=KeypointsDataset(data_list, num_keypoints=num_keypoints)
    dataloader=DataLoader(dataset, batch_size=32, shuffle=False)

    criterion=nn.MSELoss()
    total_loss=0
    num_samples=0

    with torch.no_grad():
        for inputs_2d, targets_3d in dataloader:
            inputs_2d=inputs_2d.to(device)
            targets_3d=targets_3d.to(device)

            outputs_3d=model(inputs_2d)
            loss=criterion(outputs_3d, targets_3d)

            total_loss+=loss.item() * inputs_2d.size(0)
            num_samples+=inputs_2d.size(0)

    avg_loss=total_loss / num_samples
    return avg_loss


# Haupttrainingsfunktion mit Train/Test-Split
def train_model(train_data, test_data, epochs=100, batch_size=256, learning_rate=0.002, num_keypoints=133):
    # Device setup
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets und DataLoaders erstellen
    train_dataset=KeypointsDataset(train_data, num_keypoints=num_keypoints)
    test_dataset=KeypointsDataset(test_data, num_keypoints=num_keypoints)

    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader=DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modell initialisieren
    model=model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation(num_keypoints=num_keypoints).to(device)

    # Falls vortrainiertes Modell existiert, laden
    if os.path.exists(f'net_{num_keypoints}.pth'):
        model.load_state_dict(torch.load(f'net_{num_keypoints}.pth', map_location=device))
        print('Pretrained weights loaded successfully')

    # Loss-Funktion und Optimizer
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Variablen für beste Epoche
    best_loss=float('inf')
    best_epoch=-1

    # Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss=0
        num_batches=0

        # Fortschrittsbalken für Batches
        batch_progress=tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{epochs}]', leave=False)

        for batch_idx, (inputs_2d, targets_3d) in enumerate(batch_progress):
            inputs_2d=inputs_2d.to(device)
            targets_3d=targets_3d.to(device)

            # Forward pass
            outputs_3d=model(inputs_2d)

            # Loss berechnen
            loss=criterion(outputs_3d, targets_3d)

            # Backward pass und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            num_batches+=1

            # Aktualisiere Fortschrittsbalken
            batch_progress.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

        # Durchschnittlicher Loss für die Epoche
        avg_train_loss=total_loss / num_batches

        # Evaluation auf Testdaten
        model.eval()
        test_loss=0
        with torch.no_grad():
            for inputs_2d, targets_3d in test_dataloader:
                inputs_2d=inputs_2d.to(device)
                targets_3d=targets_3d.to(device)
                outputs_3d=model(inputs_2d)
                loss=criterion(outputs_3d, targets_3d)
                test_loss+=loss.item() * inputs_2d.size(0)

        avg_test_loss=test_loss / len(test_dataset)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

        # Learning Rate anpassen basierend auf Test-Loss
        scheduler.step(avg_test_loss)

        # Prüfe ob dies die beste Epoche ist
        if avg_test_loss < best_loss:
            best_loss=avg_test_loss
            best_epoch=epoch + 1
            torch.save(model.state_dict(), f'net_{num_keypoints}_best.pth')
            print(f'New best model saved (Epoch {best_epoch}, Loss: {best_loss:.6f})')

        # Modell speichern (jede 10. Epoche)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'net_{num_keypoints}_epoch_{epoch + 1}.pth')

    # Finales Modell speichern
    torch.save(model.state_dict(), f'net_{num_keypoints}_final.pth')
    print(f'\nTraining completed!')
    print(f'Best epoch: {best_epoch} with loss: {best_loss:.6f}')
    print(f'Final model saved as net_{num_keypoints}_final.pth')
    print(f'Best model saved as net_{num_keypoints}_best.pth')

    return model


if __name__ == "__main__":
    # Daten laden
    with open('2Dto3D_train.json', 'r') as f:
        data1=json.load(f)

    with open('T3WB_v1.json', 'r') as f:
        data2=json.load(f)

    # Beide zu Listen konvertieren, falls nötig
    if isinstance(data1, dict):
        train_data=list(data1.values())
    else:
        train_data=data1.copy()

    if isinstance(data2, dict):
        train_data.extend(list(data2.values()))
    else:
        train_data.extend(data2)

    print(f"Insgesamt {len(train_data)} Einträge geladen")

    # Train/Test Split (80/20)
    train_data, test_data=train_test_split(
        train_data,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Trainingsdaten: {len(train_data)} Einträge")
    print(f"Testdaten: {len(test_data)} Einträge")

    # Trainingsparameter
    EPOCHS=175
    BATCH_SIZE=256
    LEARNING_RATE=0.002
    num_keypoints=17

    # Training starten
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=train_model(
        train_data,
        test_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_keypoints=num_keypoints
    )

    # Finale Evaluation auf Testdaten
    print("\nEvaluating model on test data...")
    test_loss=evaluate_model(model, test_data, device, num_keypoints=num_keypoints)
    print(f'Final Test Loss: {test_loss:.6f}')

    # Optional: Auch auf Trainingsdaten evaluieren
    train_loss=evaluate_model(model, train_data, device, num_keypoints=num_keypoints)
    print(f'Final Train Loss: {train_loss:.6f}')