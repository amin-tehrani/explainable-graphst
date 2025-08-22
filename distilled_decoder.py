
import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphST.model import BaseEncoder
from GraphST import GraphST, permutation
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from torch.nn.modules.module import Module


class DistilledDecoder(nn.Module):
    def __init__(self, emb_size, num_clusters):
        super(DistilledDecoder, self).__init__()
        self.emb_size = emb_size
        self.num_clusters = num_clusters
        self.layers = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_clusters)  # logits for 7 clusters
        )

        # for fitting
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()



    def forward(self, embeddings):
        # Forward pass through the linear layer
        logits = self.layers(embeddings)
        # Apply softmax to obtain probabilities for each cluster
        probs = F.softmax(logits, dim=-1)
        return probs
    
    # def _evaluate(self, graphst, base_encoder, truth, indices):
    #     """Evaluate model on given indices"""
    #     with torch.no_grad():
    #         z, _, _, _, _, _ = base_encoder(graphst.features, graphst.features_a, graphst.adj)
    #         outputs = self(z)
            
    #         # Subset to validation indices
    #         val_outputs = outputs[indices]
    #         val_truth = truth[indices]
            
    #         loss = self.criterion(val_outputs, val_truth)
            
    #         # Get predictions
    #         predictions = torch.argmax(val_outputs, dim=-1)
            
    #         # Convert to numpy for sklearn metrics
    #         val_truth_np = val_truth.cpu().numpy()
    #         predictions_np = predictions.cpu().numpy()
            
    #         # Calculate metrics
    #         accuracy = accuracy_score(val_truth_np, predictions_np)
    #         ari = adjusted_rand_score(val_truth_np, predictions_np)
    #         nmi = normalized_mutual_info_score(val_truth_np, predictions_np)
            
    #         return {
    #             'loss': loss.item(),
    #             'accuracy': accuracy,
    #             'ari': ari,
    #             'nmi': nmi
    #         }
        
    
    def fit(self, graphst: GraphST, base_encoder: BaseEncoder, labels_key='graphclust', epochs=100, lr=0.01, val_ratio=0.3, eval_frequency=10):


        print(f"Fitting distilled decoder with {labels_key=}, {epochs=}, {lr=}")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        indices = torch.randperm(graphst.adata.n_obs)
        val_size = int(graphst.adata.n_obs * val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        truth = torch.tensor(graphst.adata.obs[labels_key], device=graphst.device)
        train_truth = truth[train_indices]
        val_truth = truth[val_indices]

        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, total_steps=epochs)

        best_val_loss = float('inf')

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_ari': [],
            'val_nmi': []
        }

        self.train()
        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()

                z,_,_,_,_,_ = base_encoder(graphst.features, graphst.features_a, graphst.adj)
                outputs = self(z)

                train_outputs = outputs[train_indices]

                train_loss = self.criterion(train_outputs, train_truth)
                train_loss.backward()
                self.optimizer.step()

                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()

                history['train_loss'].append(train_loss.item())
                
                if val_ratio > 0 and (epoch + 1) % eval_frequency == 0:
                    self.eval()

                    z,_,_,_,_,_ = base_encoder(graphst.features, graphst.features_a, graphst.adj)
                    outputs = self(z)

                    val_outputs = outputs[val_indices]

                    vloss = self.criterion(val_outputs, val_truth)
        
                    # Get predictions
                    predictions = torch.argmax(val_outputs, dim=-1)
                    
                    # Convert to numpy for sklearn metrics
                    val_truth_np = val_truth.cpu().numpy()
                    predictions_np = predictions.cpu().numpy()
                    
                    # Calculate metrics
                    accuracy = accuracy_score(val_truth_np, predictions_np)
                    ari = adjusted_rand_score(val_truth_np, predictions_np)
                    nmi = normalized_mutual_info_score(val_truth_np, predictions_np)
                    
                    val_metrics = {
                        'loss': vloss.item(),
                        'accuracy': accuracy,
                        'ari': ari,
                        'nmi': nmi
                    }
                    
                    self.train()

                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                    history['val_ari'].append(val_metrics['ari'])
                    history['val_nmi'].append(val_metrics['nmi'])

                    if (vloss:=val_metrics['loss']) < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        torch.save(self.state_dict(), f'best_distilled_decoder.pt')

                    misclassification = (val_truth_np != predictions_np).sum()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}, Val Loss: {vloss:.5f}, Missclassification: {misclassification}, Val Metrics: {val_metrics}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}")
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, stopping fit")
                break

        return history
            
        
class ClusterPredictor(Module):
    def __init__(self, base_encoder: BaseEncoder, decoder: DistilledDecoder, return_probs=True):
        super(ClusterPredictor, self).__init__()
        self.base_encoder = base_encoder
        self.decoder = decoder
        self.return_probs = return_probs

    def forward(self, x, edge_index, edge_weight=None):
        # Convert edge_index to dense adjacency matrix
        adj = torch.zeros((x.size(0), x.size(0)), device=x.device)
        if edge_weight is not None:
            adj[edge_index[0], edge_index[1]] = edge_weight
        else:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        feat = x
        feat_a = permutation(feat)
        hidden_emb, _, _, _, _, _ = self.base_encoder(feat, feat_a, adj)

        probs = self.decoder(hidden_emb)

        if self.return_probs:
            return probs
        else:
            return torch.argmax(probs, dim=1)
