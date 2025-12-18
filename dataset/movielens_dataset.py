# dataset/movielens_dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional


class MovieLensDataset(Dataset):
    """
    MovieLens 20M Dataset for DQN-based Recommendation System.
    
    Each user is an environment. State = watch history + ratings.
    Action = recommend a movie. Reward = user's rating.
    """
    
    def __init__(
        self,
        data_path: str,
        min_interactions: int = 20,
        max_users: Optional[int] = None,
        use_genome: bool = False,
        seed: int = 42
    ):
        """
        Args:
            data_path: Path to MovieLens 20M data directory
            min_interactions: Minimum number of ratings per user
            max_users: Maximum number of users to load (None = all)
            use_genome: Whether to use genome tag features
            seed: Random seed
        """
        super().__init__()
        
        self.data_path = data_path
        self.min_interactions = min_interactions
        self.max_users = max_users
        self.use_genome = use_genome
        self.seed = seed
        
        np.random.seed(seed)
        
        self._load_data()
        self._preprocess()
        self._create_user_environments()
        self._build_transitions()
        
        print(f"Dataset ready: {self.num_users} users, {self.num_items} items, "
              f"{len(self.transitions)} transitions")
    
    
    def _load_data(self):
        """Load MovieLens 20M data files"""
        print("Loading MovieLens 20M...")
        
        # movie.csv
        self.movies = pd.read_csv(
            os.path.join(self.data_path, 'movie.csv'),
            dtype={'movieId': int, 'title': str, 'genres': str}
        )
        print(f"Loaded {len(self.movies)} movies")
        
        # rating.csv
        self.ratings = pd.read_csv(
            os.path.join(self.data_path, 'rating.csv'),
            dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': str}
        )
        print(f"Loaded {len(self.ratings)} ratings")
        
        # Convert timestamp
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp']).astype(int) // 10**9
        
        # Optional: genome data
        if self.use_genome:
            try:
                self.genome_scores = pd.read_csv(
                    os.path.join(self.data_path, 'genome_scores.csv'),
                    dtype={'movieId': int, 'tagId': int, 'relevance': float}
                )
                self.genome_tags = pd.read_csv(
                    os.path.join(self.data_path, 'genome_tags.csv'),
                    dtype={'tagId': int, 'tag': str}
                )
                print(f"Loaded genome: {len(self.genome_tags)} tags")
            except FileNotFoundError:
                print("Genome files not found, skipping")
                self.use_genome = False
    
    
    def _preprocess(self):
        """Preprocess data"""
        print("Preprocessing...")
        
        # Filter users
        user_counts = self.ratings.groupby('userId').size()
        valid_users = user_counts[user_counts >= self.min_interactions].index.tolist()
        
        if self.max_users is not None and len(valid_users) > self.max_users:
            valid_users = sorted(valid_users)[:self.max_users]
        
        self.ratings = self.ratings[self.ratings['userId'].isin(valid_users)]
        
        # Only keep rated movies
        rated_movies = self.ratings['movieId'].unique()
        self.movies = self.movies[self.movies['movieId'].isin(rated_movies)]
        
        # Sort by timestamp
        self.ratings = self.ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)
        
        # Parse genres
        self._parse_genres()
        
        # Create item mapping
        unique_items = sorted(self.movies['movieId'].unique())
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}
        
        self.movies['item_idx'] = self.movies['movieId'].map(self.item_id_to_idx)
        
        self.num_items = len(self.item_id_to_idx)
        self.num_users = len(valid_users)
        
        # Create item features
        self._create_item_features()
        
        print(f"Kept {self.num_users} users, {self.num_items} items")
    
    
    def _parse_genres(self):
        """Parse genres into binary vectors"""
        all_genres = set()
        for genres_str in self.movies['genres'].dropna():
            if genres_str != '(no genres listed)':
                all_genres.update(genres_str.split('|'))
        
        self.genre_list = sorted(list(all_genres))
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}
        self.num_genres = len(self.genre_list)
        
        for i, genre in enumerate(self.genre_list):
            self.movies[f'genre_{i}'] = self.movies['genres'].apply(
                lambda x: 1 if isinstance(x, str) and genre in x.split('|') else 0
            )
        
        self.genre_cols = [f'genre_{i}' for i in range(self.num_genres)]
    
    
    def _create_item_features(self):
        """Create item feature matrix [num_items, feature_dim]"""
        self.item_features = np.zeros((self.num_items, 1 + self.num_genres), dtype=np.float32)
        
        for _, row in self.movies.iterrows():
            idx = row['item_idx']
            self.item_features[idx, 0] = float(idx)
            self.item_features[idx, 1:] = row[self.genre_cols].values.astype(np.float32)
        
        if self.use_genome:
            self._add_genome_features()
    
    
    def _add_genome_features(self):
        """Add genome scores to item features"""
        genome_matrix = self.genome_scores.pivot(
            index='movieId', columns='tagId', values='relevance'
        ).fillna(0.0)
        
        num_tags = len(self.genome_tags)
        expanded = np.zeros((self.num_items, self.item_features.shape[1] + num_tags), dtype=np.float32)
        expanded[:, :self.item_features.shape[1]] = self.item_features
        
        for movie_id, idx in self.item_id_to_idx.items():
            if movie_id in genome_matrix.index:
                expanded[idx, self.item_features.shape[1]:] = genome_matrix.loc[movie_id].values
        
        self.item_features = expanded
        print(f"Added {num_tags} genome features")
    
    
    def _create_user_environments(self):
        """Create user environments"""
        self.user_envs = {}
        
        for user_id, group in self.ratings.groupby('userId'):
            interactions = []
            for _, row in group.iterrows():
                if row['movieId'] in self.item_id_to_idx:
                    interactions.append({
                        'item_idx': self.item_id_to_idx[row['movieId']],
                        'rating': row['rating'],
                        'timestamp': row['timestamp']
                    })
            
            if len(interactions) >= self.min_interactions:
                self.user_envs[user_id] = interactions
        
        self.user_ids = sorted(self.user_envs.keys())
    
    
    def _build_transitions(self):
        """Build all (s, a, r, s', done) transitions"""
        self.transitions = []
        
        for user_id in self.user_ids:
            interactions = self.user_envs[user_id]
            
            for i in range(1, len(interactions)):
                state = self._encode_state(interactions[:i])
                action_idx = interactions[i]['item_idx']
                reward = interactions[i]['rating']
                next_state = self._encode_state(interactions[:i+1])
                done = (i == len(interactions) - 1)
                
                self.transitions.append({
                    'user_id': user_id,
                    'state': state,
                    'action_idx': action_idx,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
    
    
    def _encode_state(self, history: List[Dict], max_history: int = 50) -> np.ndarray:
        """
        Fixed-size state: always returns array of length (max_history * 2)
        """
        if len(history) == 0:
            return np.zeros(max_history * 2, dtype=np.float32)
        
        # Only use last max_history items
        recent = history[-max_history:]
        
        state = []
        for h in recent:
            state.extend([float(h['item_idx']), float(h['rating'])])
        
        # Pad to EXACTLY max_history * 2
        while len(state) < max_history * 2:
            state.insert(0, 0.0)
        
        return np.array(state[:max_history * 2], dtype=np.float32)  # Ensure exact size

    def __len__(self) -> int:
        return len(self.transitions)
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (state, action, reward, next_state, done)
        """
        t = self.transitions[idx]
        
        state = torch.from_numpy(t['state'])
        action = torch.from_numpy(self.item_features[t['action_idx']])
        reward = torch.tensor(t['reward'], dtype=torch.float32)
        next_state = torch.from_numpy(t['next_state'])
        done = torch.tensor(t['done'], dtype=torch.bool)
        
        return state, action, reward, next_state, done
    
    
    def get_user_environment(self, user_id: int) -> List[Dict]:
        """Get all interactions for a user"""
        return self.user_envs[user_id]
    
    
    def get_user_transitions(self, user_id: int) -> List[int]:
        """Get transition indices for a user (for unlearning)"""
        return [i for i, t in enumerate(self.transitions) if t['user_id'] == user_id]
    
    
    def get_all_candidate_actions(self) -> torch.Tensor:
        """Get all item feature vectors [num_items, action_dim]"""
        return torch.from_numpy(self.item_features)
    
    
    def get_item_metadata(self, item_idx: int) -> Dict:
        """Get movie title and genres"""
        item_id = self.idx_to_item_id[item_idx]
        row = self.movies[self.movies['movieId'] == item_id].iloc[0]
        return {
            'item_id': item_id,
            'item_idx': item_idx,
            'title': row['title'],
            'genres': row['genres']
        }
    
    
    @property
    def action_dim(self) -> int:
        return self.item_features.shape[1]


# Usage
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    dataset = MovieLensDataset(
        data_path='data/ml-20m/',
        min_interactions=20,
        max_users=100,
        use_genome=False
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for states, actions, rewards, next_states, dones in loader:
        print(f"States: {states.shape}")
        print(f"Actions: {actions.shape}")
        print(f"Rewards: {rewards.shape}")
        break
