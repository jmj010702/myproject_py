import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class InteractionSimulator:
    def __init__(self, recipes_df, num_users=5000):
        self.recipes_df = recipes_df
        self.num_users = num_users
        self.num_recipes = len(recipes_df)
    
    def generate_users(self):
        users = []
        for user_id in range(1, self.num_users + 1):
            users.append({
                'user_id': user_id,
                'persona': np.random.choice(['health', 'quick', 'gourmet', 'random']),
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            })
        self.users_df = pd.DataFrame(users)
        print(f"✅ {self.num_users}명의 사용자 생성")
        return self.users_df
    
    def generate_interactions(self, interactions_per_user_range=(10, 50)):
        interactions = []
        for _, user in self.users_df.iterrows():
            activity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            num_interactions = int(np.random.randint(*interactions_per_user_range) * activity_multiplier[user['activity_level']])
            
            for _ in range(num_interactions):
                recipe_idx = np.random.randint(0, len(self.recipes_df))
                recipe = self.recipes_df.iloc[recipe_idx]
                
                interaction_type = np.random.choice(['view', 'click', 'like'], p=[0.6, 0.3, 0.1])
                score = {'view': 1, 'click': 2, 'like': 3}[interaction_type]
                timestamp = datetime.now() - timedelta(days=random.randint(0, 180))
                
                interactions.append({
                    'user_id': user['user_id'],
                    'recipe_id': recipe['recipe_id'],
                    'interaction_type': interaction_type,
                    'implicit_score': score,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        self.interactions_df = pd.DataFrame(interactions)
        print(f"✅ {len(interactions):,}개 상호작용 생성")
        return self.interactions_df
    
    def split_train_test(self, test_ratio=0.2):
        self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        self.interactions_df = self.interactions_df.sort_values('timestamp')
        
        train_list, test_list = [], []
        for user_id in self.interactions_df['user_id'].unique():
            user_data = self.interactions_df[self.interactions_df['user_id'] == user_id]
            split_idx = int(len(user_data) * (1 - test_ratio))
            train_list.append(user_data.iloc[:split_idx])
            test_list.append(user_data.iloc[split_idx:])
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
        return train_df, test_df
    
    def save(self, users_path, train_path, test_path):
        self.users_df.to_csv(users_path, index=False)
        train_df, test_df = self.split_train_test()
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"✅ 데이터 저장 완료")

if __name__ == "__main__":
    recipes_df = pd.read_csv('data/processed/recipes_processed.csv')
    simulator = InteractionSimulator(recipes_df, num_users=5000)
    simulator.generate_users()
    simulator.generate_interactions()
    simulator.save('data/processed/users_dummy.csv', 'data/processed/interactions_train.csv', 'data/processed/interactions_test.csv')
