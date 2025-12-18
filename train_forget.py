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


def split_dataset_by_ratio(
    dataset: MovieLensDataset,
    retain_ratio: float = 0.6,
    forget_ratio: float = 0.2,
    test_ratio: float = 0.2,
    split_by: str = 'users',
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset into Retain, Forget, and Test sets based on ratios.
    
    Args:
        dataset: MovieLens dataset
        retain_ratio: Proportion of data for retain set
        forget_ratio: Proportion of data for forget set
        test_ratio: Proportion of data for test set
        split_by: 'users' (split by users) or 'transitions' (split by transitions)
        seed: Random seed
    
    Returns:
        (retain_indices, forget_indices, test_indices)
    """
    assert abs(retain_ratio + forget_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    
    if split_by == 'users':
        # Split by users: assign each user to retain/forget/test
        all_user_ids = dataset.user_ids.copy()
        np.random.shuffle(all_user_ids)
        
        n_users = len(all_user_ids)
        n_retain = int(n_users * retain_ratio)
        n_forget = int(n_users * forget_ratio)
        
        retain_users = all_user_ids[:n_retain]
        forget_users = all_user_ids[n_retain:n_retain + n_forget]
        test_users = all_user_ids[n_retain + n_forget:]
        
        # Get all transitions for each user group
        retain_indices = []
        for uid in retain_users:
            retain_indices.extend(dataset.get_user_transitions(uid))
        
        forget_indices = []
        for uid in forget_users:
            forget_indices.extend(dataset.get_user_transitions(uid))
        
        test_indices = []
        for uid in test_users:
            test_indices.extend(dataset.get_user_transitions(uid))
        
        print(f"Dataset split by users:")
        print(f"  Retain users: {len(retain_users)} ({len(retain_indices)} transitions)")
        print(f"  Forget users: {len(forget_users)} ({len(forget_indices)} transitions)")
        print(f"  Test users: {len(test_users)} ({len(test_indices)} transitions)")
    
    elif split_by == 'transitions':
        # Split by transitions: randomly assign each transition
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        
        n_total = len(all_indices)
        n_retain = int(n_total * retain_ratio)
        n_forget = int(n_total * forget_ratio)
        
        retain_indices = all_indices[:n_retain]
        forget_indices = all_indices[n_retain:n_retain + n_forget]
        test_indices = all_indices[n_retain + n_forget:]
        
        print(f"Dataset split by transitions:")
        print(f"  Retain: {len(retain_indices)} transitions")
        print(f"  Forget: {len(forget_indices)} transitions")
        print(f"  Test: {len(test_indices)} transitions")
    
    else:
        raise ValueError(f"Unknown split_by: {split_by}. Choose 'users' or 'transitions'")
    
    return retain_indices, forget_indices, test_indices


def split_dataset_by_user_list(
    dataset: MovieLensDataset,
    forget_user_ids: List[int],
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset by specifying exact users to forget.
    
    Args:
        dataset: MovieLens dataset
        forget_user_ids: List of specific user IDs to forget
        test_ratio: Proportion of transitions to use as test (sampled from all users)
        seed: Random seed
    
    Returns:
        (retain_indices, forget_indices, test_indices)
    """
    np.random.seed(seed)
    
    all_user_ids = dataset.user_ids
    retain_user_ids = [uid for uid in all_user_ids if uid not in forget_user_ids]
    
    # Get all transitions for each group
    forget_transitions = []
    for uid in forget_user_ids:
        forget_transitions.extend(dataset.get_user_transitions(uid))
    
    retain_transitions = []
    for uid in retain_user_ids:
        retain_transitions.extend(dataset.get_user_transitions(uid))
    
    # Shuffle
    np.random.shuffle(retain_transitions)
    np.random.shuffle(forget_transitions)
    
    # Split both retain and forget into train/test
    retain_test_size = int(len(retain_transitions) * test_ratio)
    forget_test_size = int(len(forget_transitions) * test_ratio)
    
    test_indices = (
        retain_transitions[:retain_test_size] + 
        forget_transitions[:forget_test_size]
    )
    
    retain_train_indices = retain_transitions[retain_test_size:]
    forget_train_indices = forget_transitions[forget_test_size:]
    
    print(f"Dataset split by user list:")
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


def main(config_override: Dict = None):
    # ============================================
    # CONFIGURATION
    # ============================================
    config = {
        # Dataset
        'data_path': '../data_movie/',
        'max_users': 2500,
        'min_interactions': 20,
        
        # Split configuration - CHOOSE ONE METHOD:
        # Method 1: Split by ratio
        'split_method': 'ratio',  # 'ratio' or 'user_list'
        'retain_ratio': 0.6,
        'forget_ratio': 0.2,
        'test_ratio': 0.2,
        'split_by': 'users',  # 'users' or 'transitions'
        
        # Method 2: Split by user list (only used if split_method='user_list')
        'forget_user_ids': [1, 2, 3, 4, 5],
        'test_ratio_user_list': 0.2,
        
        # Training
        'batch_size': 128,
        'train_epochs': 10,
        'unlearn_epochs': 10,
        'learning_rate': 1e-4,
        'hidden_dim': 256,
        'gamma': 0.99,
        
        # Unlearning
        'unlearn_lr': 1e-4,
        'lambda_retain': 1.0,
        'grad_clip': 1.0,
        
        # Other
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints/'
    }

    if config_override:
        config.update(config_override)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ============================================
    # LOAD DATASET
    # ============================================
    print("Loading dataset...")
    dataset = MovieLensDataset(
        data_path=config['data_path'],
        min_interactions=config['min_interactions'],
        max_users=config['max_users'],
        use_genome=False,
        seed=config['seed']
    )
    
    # ============================================
    # SPLIT DATASET
    # ============================================
    if config['split_method'] == 'ratio':
        retain_indices, forget_indices, test_indices = split_dataset_by_ratio(
            dataset=dataset,
            retain_ratio=config['retain_ratio'],
            forget_ratio=config['forget_ratio'],
            test_ratio=config['test_ratio'],
            split_by=config['split_by'],
            seed=config['seed']
        )
    elif config['split_method'] == 'user_list':
        retain_indices, forget_indices, test_indices = split_dataset_by_user_list(
            dataset=dataset,
            forget_user_ids=config['forget_user_ids'],
            test_ratio=config['test_ratio_user_list'],
            seed=config['seed']
        )
    else:
        raise ValueError(f"Unknown split_method: {config['split_method']}")
    
    loaders = create_dataloaders(
        dataset, retain_indices, forget_indices, test_indices, config['batch_size']
    )
    candidate_items = dataset.get_all_candidate_actions()
    
    # Store all results
    all_results = {}
    
    # ============================================
    # PHASE 1: BASELINE (Retain only)
    # ============================================
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (Retain only)")
    print("="*60)
    
    agent_baseline = DQNAgent(
        state_dim=100,
        action_dim=dataset.action_dim,
        hidden_dim=config['hidden_dim'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        device=config['device']
    )
    
    for epoch in range(config['train_epochs']):
        loss = train_dqn_epoch(agent_baseline, loaders['retain'], config['device'])
        print(f"Epoch {epoch+1}/{config['train_epochs']} - Loss: {loss:.4f}")
    
    test_baseline = evaluate_recommender(
        agent_baseline.q_network, loaders['test'], candidate_items, config['device']
    )
    forget_baseline = evaluate_recommender(
        agent_baseline.q_network, loaders['forget'], candidate_items, config['device']
    )
    print_evaluation_results("BASELINE", test_baseline, forget_baseline)
    
    agent_baseline.save(os.path.join(config['save_dir'], 'baseline.pt'))
    all_results['baseline'] = {'test': test_baseline, 'forget': forget_baseline}
    
    # ============================================
    # PHASE 2: AFTER LEARNING (Retain + Forget)
    # ============================================
    print("\n" + "="*60)
    print("PHASE 2: AFTER LEARNING (Retain + Forget)")
    print("="*60)
    
    agent_learned = DQNAgent(
        state_dim=100,
        action_dim=dataset.action_dim,
        hidden_dim=config['hidden_dim'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        device=config['device']
    )
    
    for epoch in range(config['train_epochs']):
        loss = train_dqn_epoch(agent_learned, loaders['retain+forget'], config['device'])
        print(f"Epoch {epoch+1}/{config['train_epochs']} - Loss: {loss:.4f}")
    
    test_learned = evaluate_recommender(
        agent_learned.q_network, loaders['test'], candidate_items, config['device']
    )
    forget_learned = evaluate_recommender(
        agent_learned.q_network, loaders['forget'], candidate_items, config['device']
    )
    print_evaluation_results("AFTER LEARNING", test_learned, forget_learned)
    
    agent_learned.save(os.path.join(config['save_dir'], 'after_learning.pt'))
    all_results['after_learning'] = {'test': test_learned, 'forget': forget_learned}
    
    # ============================================
    # PHASE 3: AFTER UNLEARNING (Decremental RL)
    # ============================================
    print("\n" + "="*60)
    print("PHASE 3: AFTER UNLEARNING (Decremental RL)")
    print("="*60)
    
    q_unlearned = decremental_unlearn(
        q_network=agent_learned.q_network,
        forget_loader=loaders['forget'],
        retain_loader=loaders['retain'],
        candidate_items=candidate_items,
        epochs=config['unlearn_epochs'],
        lr=config['unlearn_lr'],
        lambda_retain=config['lambda_retain'],
        grad_clip=config['grad_clip'],
        device=config['device']
    )
    
    test_unlearned = evaluate_recommender(
        q_unlearned, loaders['test'], candidate_items, config['device']
    )
    forget_unlearned = evaluate_recommender(
        q_unlearned, loaders['forget'], candidate_items, config['device']
    )
    print_evaluation_results("AFTER UNLEARNING", test_unlearned, forget_unlearned)
    
    torch.save(q_unlearned.state_dict(), os.path.join(config['save_dir'], 'after_unlearning.pt'))
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
    results_to_save = {
        'config': config,
        'baseline': all_results['baseline'],
        'after_learning': all_results['after_learning'],
        'after_unlearning': all_results['after_unlearning'],
        'summary': summary
    }
    
    with open(os.path.join(config['save_dir'], 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to {config['save_dir']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
