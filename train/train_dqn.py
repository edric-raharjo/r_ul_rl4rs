# train/train_dqn.py
import torch
from torch.utils.data import DataLoader, Subset
from dataset.movielens_dataset import MovieLensDataset
from model.dqn_agent import DQNAgent
from model.dqn_model import QNetwork
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Tuple


def split_dataset(
    dataset: MovieLensDataset,
    forget_user_ids: List[int],
    test_ratio: float = 0.2
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset into Retain, Forget, and Test sets.
    
    Args:
        dataset: Full MovieLens dataset
        forget_user_ids: List of user IDs to forget
        test_ratio: Ratio of transitions to use as test set (from all users)
    
    Returns:
        (retain_indices, forget_indices, test_indices)
    """
    all_user_ids = dataset.user_ids
    retain_user_ids = [uid for uid in all_user_ids if uid not in forget_user_ids]
    
    # Get all transitions for each group
    forget_transitions = []
    for uid in forget_user_ids:
        forget_transitions.extend(dataset.get_user_transitions(uid))
    
    retain_transitions = []
    for uid in retain_user_ids:
        retain_transitions.extend(dataset.get_user_transitions(uid))
    
    # Split both retain and forget into train/test
    np.random.shuffle(retain_transitions)
    np.random.shuffle(forget_transitions)
    
    # Test set: sample from both retain and forget
    retain_test_size = int(len(retain_transitions) * test_ratio)
    forget_test_size = int(len(forget_transitions) * test_ratio)
    
    test_indices = (
        retain_transitions[:retain_test_size] + 
        forget_transitions[:forget_test_size]
    )
    
    # Training sets
    retain_train_indices = retain_transitions[retain_test_size:]
    forget_train_indices = forget_transitions[forget_test_size:]
    
    print(f"Dataset split:")
    print(f"  Retain train: {len(retain_train_indices)} transitions")
    print(f"  Forget train: {len(forget_train_indices)} transitions")
    print(f"  Test (both): {len(test_indices)} transitions")
    
    return retain_train_indices, forget_train_indices, test_indices


def create_dataloaders(
    dataset: MovieLensDataset,
    retain_indices: List[int],
    forget_indices: List[int],
    test_indices: List[int],
    batch_size: int = 128
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for retain, forget, and test sets.
    
    Returns:
        Dictionary with keys: 'retain', 'forget', 'test', 'retain+forget'
    """
    retain_loader = DataLoader(
        Subset(dataset, retain_indices),
        batch_size=batch_size,
        shuffle=True
    )
    
    forget_loader = DataLoader(
        Subset(dataset, forget_indices),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Combined retain+forget for "after learning" phase
    combined_loader = DataLoader(
        Subset(dataset, retain_indices + forget_indices),
        batch_size=batch_size,
        shuffle=True
    )
    
    return {
        'retain': retain_loader,
        'forget': forget_loader,
        'test': test_loader,
        'retain+forget': combined_loader
    }


def train_dqn_epoch(
    agent: DQNAgent,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Train DQN for one epoch on given dataloader.
    
    Returns:
        Average loss for the epoch
    """
    losses = []
    
    for batch in tqdm(dataloader, desc="Training"):
        states, actions, rewards, next_states, dones = batch
        
        # Add to replay buffer
        for i in range(len(states)):
            agent.replay_buffer.push(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
        
        # Train step
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
    
    return np.mean(losses) if losses else 0.0


def evaluate_dqn(
    q_network: QNetwork,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate Q-network on test set.
    
    Returns:
        Dictionary with metrics: 'mse', 'mae'
    """
    q_network.eval()
    mse_losses = []
    mae_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            states, actions, rewards, next_states, dones = batch
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            
            # Predicted Q-values
            q_pred = q_network(states, actions)
            
            # Use reward as target (simplified evaluation)
            mse = torch.mean((q_pred - rewards) ** 2).item()
            mae = torch.mean(torch.abs(q_pred - rewards)).item()
            
            mse_losses.append(mse)
            mae_losses.append(mae)
    
    q_network.train()
    
    return {
        'mse': np.mean(mse_losses),
        'mae': np.mean(mae_losses)
    }


def decremental_unlearn(
    q_network: QNetwork,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    candidate_items: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-4,
    lambda_retain: float = 1.0,
    grad_clip: float = 1.0,
    device: str = 'cuda'
) -> QNetwork:
    """
    Decremental RL unlearning for MovieLens DQN.
    
    Forget loss: Minimize Q-values on random actions from forget states
    Retain loss: Preserve Q-values on retain data
    
    Args:
        q_network: Pre-trained Q-network to unlearn from
        forget_loader: DataLoader with forget user transitions
        retain_loader: DataLoader with retain user transitions
        candidate_items: All candidate item features [num_items, action_dim]
        epochs: Number of unlearning epochs
        lr: Learning rate
        lambda_retain: Weight for retain loss
        grad_clip: Gradient clipping threshold
        device: Device
    
    Returns:
        Unlearned Q-network
    """
    print("\n=== Starting Decremental RL Unlearning ===")
    
    # 1. Create frozen reference network
    q_ref = copy.deepcopy(q_network).to(device)
    q_ref.eval()
    for p in q_ref.parameters():
        p.requires_grad_(False)
    
    # 2. Trainable current network
    q_cur = q_network.to(device)
    q_cur.train()
    for p in q_cur.parameters():
        p.requires_grad_(True)
    
    optimizer = torch.optim.Adam(q_cur.parameters(), lr=lr)
    
    # 3. Move candidate items to device
    candidate_items = candidate_items.to(device)
    num_items = len(candidate_items)
    
    # 4. Cycle through retain loader
    retain_iter = iter(retain_loader)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_forget_losses = []
        epoch_retain_losses = []
        
        for batch_idx, forget_batch in enumerate(tqdm(forget_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # ===== FORGET BATCH =====
            states_f, actions_f, rewards_f, next_states_f, dones_f = forget_batch
            states_f = states_f.to(device)
            B_f = len(states_f)
            
            # Sample random items for each state
            random_indices = torch.randint(0, num_items, (B_f,), device=device)
            random_items = candidate_items[random_indices]  # [B_f, action_dim]
            
            # Compute Q(s_forget, random_item)
            q_forget = q_cur(states_f, random_items)  # [B_f]
            
            # Forget loss: minimize Q on random policy
            L_forget = q_forget.mean()
            
            # ===== RETAIN BATCH =====
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            
            states_r, actions_r, rewards_r, next_states_r, dones_r = retain_batch
            states_r = states_r.to(device)
            actions_r = actions_r.to(device)
            
            # Current Q on retain data
            q_retain_cur = q_cur(states_r, actions_r)  # [B_r]
            
            # Reference Q on retain data
            with torch.no_grad():
                q_retain_ref = q_ref(states_r, actions_r)  # [B_r]
            
            # Retain loss: preserve Q-values
            L_retain = torch.abs(q_retain_cur - q_retain_ref).mean()
            
            # ===== TOTAL LOSS =====
            loss = L_forget + lambda_retain * L_retain
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(q_cur.parameters(), grad_clip)
            
            optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            epoch_forget_losses.append(L_forget.item())
            epoch_retain_losses.append(L_retain.item())
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_forget = np.mean(epoch_forget_losses)
        avg_retain = np.mean(epoch_retain_losses)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | "
              f"L_forget: {avg_forget:.4f} | L_retain: {avg_retain:.4f}")
    
    print("=== Unlearning Complete ===\n")
    return q_cur


def main():
    """
    Main training pipeline with baseline, learning, and unlearning phases.
    """
    # Configuration
    data_path = 'data/ml-20m/'
    max_users = 1000
    forget_user_ids = [1, 2, 3, 4, 5]  # Users to forget
    test_ratio = 0.2
    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset
    print("Loading dataset...")
    dataset = MovieLensDataset(
        data_path=data_path,
        min_interactions=20,
        max_users=max_users,
        use_genome=False,
        seed=42
    )
    
    # Split dataset
    retain_indices, forget_indices, test_indices = split_dataset(
        dataset, forget_user_ids, test_ratio
    )
    
    # Create dataloaders
    loaders = create_dataloaders(
        dataset, retain_indices, forget_indices, test_indices, batch_size
    )
    
    # Get candidate items
    candidate_items = dataset.get_all_candidate_actions()
    
    # ============================================
    # PHASE 1: BASELINE (Train only on Retain)
    # ============================================
    print("\n" + "="*50)
    print("PHASE 1: BASELINE (Retain only)")
    print("="*50)
    
    agent_baseline = DQNAgent(
        state_dim=100,
        action_dim=dataset.action_dim,
        hidden_dim=256,
        learning_rate=1e-4,
        gamma=0.99,
        device=device
    )
    
    # Train on retain only
    for epoch in range(10):
        loss = train_dqn_epoch(agent_baseline, loaders['retain'], device)
        print(f"Epoch {epoch+1}/10 - Loss: {loss:.4f}")
    
    # Evaluate
    test_metrics = evaluate_dqn(agent_baseline.q_network, loaders['test'], device)
    forget_metrics = evaluate_dqn(agent_baseline.q_network, loaders['forget'], device)
    print(f"Baseline - Test: {test_metrics}, Forget: {forget_metrics}")
    
    agent_baseline.save('checkpoints/baseline.pt')
    
    # ============================================
    # PHASE 2: AFTER LEARNING (Retain + Forget)
    # ============================================
    print("\n" + "="*50)
    print("PHASE 2: AFTER LEARNING (Retain + Forget)")
    print("="*50)
    
    agent_learned = DQNAgent(
        state_dim=100,
        action_dim=dataset.action_dim,
        hidden_dim=256,
        learning_rate=1e-4,
        gamma=0.99,
        device=device
    )
    
    # Train on retain + forget
    for epoch in range(10):
        loss = train_dqn_epoch(agent_learned, loaders['retain+forget'], device)
        print(f"Epoch {epoch+1}/10 - Loss: {loss:.4f}")
    
    # Evaluate
    test_metrics = evaluate_dqn(agent_learned.q_network, loaders['test'], device)
    forget_metrics = evaluate_dqn(agent_learned.q_network, loaders['forget'], device)
    print(f"After Learning - Test: {test_metrics}, Forget: {forget_metrics}")
    
    agent_learned.save('checkpoints/after_learning.pt')
    
    # ============================================
    # PHASE 3: AFTER UNLEARNING (Decremental RL)
    # ============================================
    print("\n" + "="*50)
    print("PHASE 3: AFTER UNLEARNING (Decremental RL)")
    print("="*50)
    
    # Start from learned model
    q_unlearned = decremental_unlearn(
        q_network=agent_learned.q_network,
        forget_loader=loaders['forget'],
        retain_loader=loaders['retain'],
        candidate_items=candidate_items,
        epochs=10,
        lr=1e-4,
        lambda_retain=1.0,
        device=device
    )
    
    # Evaluate
    test_metrics = evaluate_dqn(q_unlearned, loaders['test'], device)
    forget_metrics = evaluate_dqn(q_unlearned, loaders['forget'], device)
    print(f"After Unlearning - Test: {test_metrics}, Forget: {forget_metrics}")
    
    # Save unlearned model
    torch.save(q_unlearned.state_dict(), 'checkpoints/after_unlearning.pt')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == '__main__':
    main()
