from model import BrainGraphClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, input_dim, num_nodes_wm, num_nodes_gm, device, 
             learning_rate=0.01, weight_decay=1e-4, dropout=0.0, layers=2,
             hidden_dim=64, adv_lambda=0.1, knn_lambda=0.1, k=5, savepath='ckpt/'):
        self.knn_lambda = knn_lambda
        # self.knn_module = KNNModule(k=k, num_nodes_wm=num_nodes_wm, num_nodes_gm=num_nodes_gm,device = device)
        self.adv_lambda = adv_lambda
        self.num_nodes_wm = num_nodes_wm
        self.num_nodes_gm = num_nodes_gm
        self.device = torch.device(device)
        self.model = BrainGraphClassifier(
            input_dim=input_dim, 
            hidden_dim= hidden_dim,
            num_nodes_wm=num_nodes_wm, 
            num_nodes_gm=num_nodes_gm,
            dropout=dropout,
            layers=layers
        ).to(self.device)
        
        self.optimizer = optim.AdamW( 
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_acc = 0
        self.patience = 40
        self.patience_counter = 0

        self.savepath = savepath

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            logits, adv_loss, wm_features, gm_features = self.model(batch)
            
            gm_embeddings = self.model.get_gm_embeddings()  
            
            knn_loss = self.knn_module.forward(
                gm_embeddings,
                batch['aa_index']
            )
            
            labels = batch['label'].view(-1)
            cls_loss = self.criterion(logits, labels)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            loss = cls_loss + self.adv_lambda  * adv_loss
            # loss = cls_loss + self.adv_lambda * adv_loss + self.knn_lambda * knn_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)           
            self.optimizer.step()         
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, dataloader):
        self.model.eval()
        predictions = []
        true_labels = []
        test_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                logits, adv_loss, wm_features, gm_features = self.model(batch)

                pred = logits.argmax(dim=1).cpu().numpy()
                labels = batch['label'].view(-1).cpu().numpy()
                test_loss += self.criterion(logits, batch['label'].view(-1)).item()
                predictions.extend(pred)
                true_labels.extend(labels)

                
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        test_loss /= len(dataloader)
        
        cm = confusion_matrix(true_labels, predictions)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()  
        else:
            tn = fp = fn = tp = 0

        unique_classes = set(true_labels)
        if len(unique_classes) < 2:
            auc = 0.5  
        else:
            auc = roc_auc_score(true_labels, predictions)
        
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        f1 = f1_score(true_labels, predictions) if len(set(predictions)) > 1 else 0  # 防止只有一个类的情况

        metrics = {
            'acc': accuracy_score(true_labels, predictions),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'auc': auc,
        }
        
        return test_loss, metrics

    def train_and_evaluate(self, train_loader, val_loader, test_loader, epochs):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            val_acc = val_metrics['acc']
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            self.scheduler.step(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                if not os.path.exists(self.savepath):
                    os.makedirs(self.savepath)
                torch.save(self.model.state_dict(), f'{self.savepath}/best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break

        self.model.load_state_dict(torch.load(f'{self.savepath}/best_model.pth'))
        test_loss, test_metrics = self.evaluate(test_loader)
        print(f'Test Acc: {test_metrics["acc"]:.2%}, Test loss: {test_loss:.4f}')
        
        return test_metrics, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
        }
