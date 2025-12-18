from torch.utils.data import DataLoader
from dataset.movielens_dataset import MovieLensDataset

dataset = MovieLensDataset(
    data_path=r'E:\Kuliah\Kuliah\Kuliah\PRODI\Semester 7\ProSkripCode\data_movie',
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