# train/train_forget.py
import torch
from torch.utils.data import DataLoader, Subset
from dataset.movielens_dataset import MovieLensDataset
from model.dqn_agent import DQNAgent
from model.dqn_model import QNetwork
from eval.evaluate import (
    evaluate_recommender,
    print_evaluation_results,
    compute_unlearning_summary,
    print_unlearning_summary
)
from tqdm import tqdm
import numpy as np
import copy
from typing import List, Dict, Tuple
import os
import json


def split_dataset(
    dataset: MovieLensDataset,
    forget_user_ids: List[int],
    test_ratio: float = 0.2
) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset into Retain, Forget, and Test sets."""
    all_user_ids = dataset.user_ids
    retain_user_ids = [uid for uid in all_user_ids if uid not in forget_user_ids]
    
    forget_transitions = []
    for uid in forget_user_ids:
        forget_transitions.extend(dataset.get_user_transitions(uid))
    
    retain_transitions = []
    for uid in retain_user_ids:
        retain_transitions.extend(dataset.get_user_transitions(uid))
    
    np.random.shuffle(retain_transitions)
    np.random.shuffle(forget_transitions)
    
    retain_test_size = int(len(retain_transitions) * test_ratio)
    forget_test_size = int(len(forget_transitions) * test_ratio)
    
    test_indices = (
        retain_transitions[:retain_test_size] + 
        forget_transitions[:forget_test_size]
    )
    
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
    """Create dataloaders for all splits."""
    return {
        'retain': DataLoader(Subset(dataset, retain_indices), batch_size=batch_size, shuffle=True),
        'forget': DataLoader(Subset(dataset, forget_indices), batch_size=batch_size, shuffle=True),
        'test': DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False),
        'retain+forget': DataLoader(Subset(dataset, retain_indices + forget_indices), batch_size=batch_size, shuffle=True)
    }


def train_dqn_epoch(agent: DQNAgent, dataloader: DataLoader, device: str = 'cuda') -> float:
    """Train DQN for one epoch."""
    losses = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        states, actions, rewards, next_states, dones = batch
        
        for i in range(len(states)):
            agent.replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
    
    return np.mean(losses) if losses else 0.0


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
    """Decremental RL unlearning."""
    print("\n=== Starting Decremental RL Unlearning ===")
    
    q_ref = copy.deepcopy(q_network).to(device)
    q_ref.eval()
    for p in q_ref.parameters():
        p.requires_grad_(False)
    
    q_cur = q_network.to(device)
    q_cur.train()
    
    optimizer = torch.optim.Adam(q_cur.parameters(), lr=lr)
    candidate_items = candidate_items.to(device)
    num_items = len(candidate_items)
    
    retain_iter = iter(retain_loader)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_forget_losses = []
        epoch_retain_losses = []
        
        for batch_idx, forget_batch in enumerate(tqdm(forget_loader, desc=f"Unlearning Epoch {epoch+1}/{epochs}", leave=False)):
            states_f, actions_f, rewards_f, next_states_f, dones_f = forget_batch
            states_f = states_f.to(device)
            B_f = len(states_f)
            
            random_indices = torch.randint(0, num_items, (B_f,), device=device)
            random_items = candidate_items[random_indices]
            
            q_forget = q_cur(states_f, random_items)
            L_forget = q_forget.mean()
            
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            
            states_r, actions_r, rewards_r, next_states_r, dones_r = retain_batch
            states_r = states_r.to(device)
            actions_r = actions_r.to(device)
            
            q_retain_cur = q_cur(states_r, actions_r)
            
            with torch.no_grad():
                q_retain_ref = q_ref(states_r, actions_r)
            
            L_retain = torch.abs(q_retain_cur - q_retain_ref).mean()
            
            loss = L_forget + lambda_retain * L_retain
            
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(q_cur.parameters(), grad_clip)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_forget_losses.append(L_forget.item())
            epoch_retain_losses.append(L_retain.item())
        
        avg_loss = np.mean(epoch_losses)
        avg_forget = np.mean(epoch_forget_losses)
        avg_retain = np.mean(epoch_retain_losses)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | L_forget: {avg_forget:.4f} | L_retain: {avg_retain:.4f}")
    
    print("=== Unlearning Complete ===\n")
    return q_cur


def main():
    # Configuration
    data_path = '../data_movie/'
    max_users = 1000
    forget_user_ids = [1, 2, 3, 4, 5]
    test_ratio = 0.2
    batch_size = 128
    train_epochs = 5
    unlearn_epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MovieLensDataset(
        data_path=data_path,
        min_interactions=20,
        max_users=max_users,
        use_genome=False,
        seed=42
    )
    
    retain_indices, forget_indices, test_indices = split_dataset(dataset, forget_user_ids, test_ratio)
    loaders = create_dataloaders(dataset, retain_indices, forget_indices, test_indices, batch_size)
    candidate_items = dataset.get_all_candidate_actions()
    
    # Store all results
    all_results = {}
    
    # ============================================
    # PHASE 1: BASELINE
    # ============================================
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (Retain only)")
    print("="*60)
    
    agent_baseline = DQNAgent(
        state_dim=100,
        action_dim=dataset.action_dim,
        hidden_dim=256,
        learning_rate=1e-4,
        gamma=0.99,
        device=device
    )
    
    for epoch in range(train_epochs):
        loss = train_dqn_epoch(agent_baseline, loaders['retain'], device)
        print(f"Epoch {epoch+1}/{train_epochs} - Loss: {loss:.4f}")
    
    test_baseline = evaluate_recommender(agent_baseline.q_network, loaders['test'], candidate_items, device)
    forget_baseline = evaluate_recommender(agent_baseline.q_network, loaders['forget'], candidate_items, device)
    print_evaluation_results("BASELINE", test_baseline, forget_baseline)
    
    agent_baseline.save(os.path.join(save_dir, 'baseline.pt'))
    all_results['baseline'] = {'test': test_baseline, 'forget': forget_baseline}
    
    # ============================================
    # PHASE 2: AFTER LEARNING
    # ============================================
    print("\n" + "="*60)
    print("PHASE 2: AFTER LEARNING (Retain + Forget)")
    print("="*60)
    
    agent_learned = DQNAgent(
        state_dim=100,
        action_dim=dataset.action_dim,
        hidden_dim=256,
        learning_rate=1e-4,
        gamma=0.99,
        device=device
    )
    
    for epoch in range(train_epochs):
        loss = train_dqn_epoch(agent_learned, loaders['retain+forget'], device)
        print(f"Epoch {epoch+1}/{train_epochs} - Loss: {loss:.4f}")
    
    test_learned = evaluate_recommender(agent_learned.q_network, loaders['test'], candidate_items, device)
    forget_learned = evaluate_recommender(agent_learned.q_network, loaders['forget'], candidate_items, device)
    print_evaluation_results("AFTER LEARNING", test_learned, forget_learned)
    
    agent_learned.save(os.path.join(save_dir, 'after_learning.pt'))
    all_results['after_learning'] = {'test': test_learned, 'forget': forget_learned}
    
    # ============================================
    # PHASE 3: AFTER UNLEARNING
    # ============================================
    print("\n" + "="*60)
    print("PHASE 3: AFTER UNLEARNING (Decremental RL)")
    print("="*60)
    
    q_unlearned = decremental_unlearn(
        q_network=agent_learned.q_network,
        forget_loader=loaders['forget'],
        retain_loader=loaders['retain'],
        candidate_items=candidate_items,
        epochs=unlearn_epochs,
        lr=1e-4,
        lambda_retain=1.0,
        device=device
    )
    
    test_unlearned = evaluate_recommender(q_unlearned, loaders['test'], candidate_items, device)
    forget_unlearned = evaluate_recommender(q_unlearned, loaders['forget'], candidate_items, device)
    print_evaluation_results("AFTER UNLEARNING", test_unlearned, forget_unlearned)
    
    torch.save(q_unlearned.state_dict(), os.path.join(save_dir, 'after_unlearning.pt'))
    all_results['after_unlearning'] = {'test': test_unlearned, 'forget': forget_unlearned}
    
    # ============================================
    # SUMMARY
    # ============================================
    summary = compute_unlearning_summary(
        test_baseline, test_learned, test_unlearned,
        forget_baseline, forget_learned, forget_unlearned
    )
    print_unlearning_summary(summary)
    
    # Save all results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump({
            'baseline': all_results['baseline'],
            'after_learning': all_results['after_learning'],
            'after_unlearning': all_results['after_unlearning'],
            'summary': summary
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
