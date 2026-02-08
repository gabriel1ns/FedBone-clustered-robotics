import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class GPAggregation:
    """
    Gradient Projection Aggregation with Historical Gradient Attention
    
    Implements Algorithm 2 from FedBone paper:
    1. Scale gradients using attention to historical gradients
    2. Project conflicting gradients onto normal planes
    3. Average de-conflicted gradients
    """
    
    def __init__(self, gradient_dim):
        self.gradient_dim = gradient_dim
        self.previous_aggregated_grad = None
        
    def flatten_gradients(self, model_grads):
        """Flatten model gradients to 1D vector"""
        flat_grads = []
        for grad in model_grads.values():
            if grad is not None:
                flat_grads.append(grad.flatten())
        return torch.cat(flat_grads)
    
    def unflatten_gradients(self, flat_grad, model_grads_template):
        """Unflatten 1D gradient back to model structure"""
        unflat_grads = OrderedDict()
        idx = 0
        for key, grad in model_grads_template.items():
            if grad is not None:
                numel = grad.numel()
                unflat_grads[key] = flat_grad[idx:idx+numel].view(grad.shape)
                idx += numel
            else:
                unflat_grads[key] = None
        return unflat_grads
    
    def compute_gradient_attention(self, current_grad, historical_grad):
        """
        Compute attention score between current and historical gradient
        Using softmax of dot product similarity
        """
        if historical_grad is None:
            return 1.0
        
        # Normalize gradients
        current_norm = F.normalize(current_grad.unsqueeze(0), p=2, dim=1)
        historical_norm = F.normalize(historical_grad.unsqueeze(0), p=2, dim=1)
        
        # Compute attention score
        similarity = torch.mm(current_norm, historical_norm.t()).squeeze()
        attention = torch.softmax(similarity.unsqueeze(0), dim=0).squeeze()
        
        return attention.item()
    
    def project_conflicting_gradients(self, grad_i, grad_j):
        """
        Project gradient i onto normal plane of gradient j if they conflict
        
        Implements equation (2) from FedBone paper:
        ∇'_i = ∇_i - (∇_i · ∇_j / ||∇_j||²) * ∇_j
        """
        dot_product = torch.dot(grad_i, grad_j)
        
        # Check if gradients conflict (negative dot product)
        if dot_product < 0:
            # Project grad_i onto normal plane of grad_j
            projection = (dot_product / (torch.norm(grad_j) ** 2)) * grad_j
            grad_i_projected = grad_i - projection
            return grad_i_projected
        else:
            # No conflict, return original
            return grad_i
    
    def aggregate(self, client_gradients, client_sizes=None):
        """
        Main aggregation function implementing Algorithm 2
        
        Args:
            client_gradients: List of gradient dictionaries from each client
            client_sizes: List of dataset sizes for weighted aggregation
            
        Returns:
            Aggregated gradient dictionary
        """
        num_clients = len(client_gradients)
        
        if client_sizes is None:
            client_sizes = [1] * num_clients
        
        # Get template for gradient structure
        grad_template = client_gradients[0]
        
        # Flatten all client gradients
        flat_grads = []
        for client_grad in client_gradients:
            flat_grad = self.flatten_gradients(client_grad)
            flat_grads.append(flat_grad)
        
        # Step 1: Scale gradients using attention to historical gradients
        scaled_grads = []
        for i, flat_grad in enumerate(flat_grads):
            attention = self.compute_gradient_attention(
                flat_grad, 
                self.previous_aggregated_grad
            )
            scaled_grad = attention * flat_grad
            scaled_grads.append(scaled_grad)
        
        # Step 2: Project conflicting gradients
        projected_grads = []
        for i in range(num_clients):
            grad_i = scaled_grads[i].clone()
            
            # Project onto normal planes of all other clients
            for j in range(num_clients):
                if i != j:
                    grad_j = scaled_grads[j]
                    grad_i = self.project_conflicting_gradients(grad_i, grad_j)
            
            projected_grads.append(grad_i)
        
        # Step 3: Weighted average aggregation
        total_size = sum(client_sizes)
        aggregated_flat = torch.zeros_like(projected_grads[0])
        
        for i, proj_grad in enumerate(projected_grads):
            weight = client_sizes[i] / total_size
            aggregated_flat += weight * proj_grad
        
        # Store for next round
        self.previous_aggregated_grad = aggregated_flat.clone().detach()
        
        # Unflatten back to model structure
        aggregated_grads = self.unflatten_gradients(aggregated_flat, grad_template)
        
        return aggregated_grads
    
    def reset_history(self):
        """Reset historical gradient (e.g., when re-clustering)"""
        self.previous_aggregated_grad = None


def compute_gradient_conflict_score(client_gradients):
    """
    Compute average gradient conflict score between clients
    Useful for monitoring training dynamics
    """
    num_clients = len(client_gradients)
    if num_clients < 2:
        return 0.0
    
    # Flatten gradients
    flat_grads = []
    for grad_dict in client_gradients:
        flat_grad = []
        for grad in grad_dict.values():
            if grad is not None:
                flat_grad.append(grad.flatten())
        flat_grads.append(torch.cat(flat_grad))
    
    # Compute pairwise conflicts
    conflicts = []
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            grad_i = F.normalize(flat_grads[i].unsqueeze(0), p=2, dim=1)
            grad_j = F.normalize(flat_grads[j].unsqueeze(0), p=2, dim=1)
            
            similarity = torch.mm(grad_i, grad_j.t()).squeeze()
            
            # Conflict when similarity < 0
            if similarity < 0:
                conflicts.append(abs(similarity.item()))
    
    if len(conflicts) == 0:
        return 0.0
    
    return np.mean(conflicts)


def simple_average_aggregation(client_gradients, client_sizes=None):
    """
    Simple weighted average aggregation (baseline)
    """
    num_clients = len(client_gradients)
    
    if client_sizes is None:
        client_sizes = [1] * num_clients
    
    total_size = sum(client_sizes)
    aggregated = OrderedDict()
    
    # Get template
    grad_template = client_gradients[0]
    
    for key in grad_template.keys():
        aggregated[key] = torch.zeros_like(grad_template[key])
        
        for i, client_grad in enumerate(client_gradients):
            if client_grad[key] is not None:
                weight = client_sizes[i] / total_size
                aggregated[key] += weight * client_grad[key]
    
    return aggregated