"""
모든 필수 파일을 자동으로 생성하는 스크립트
"""

import os

def create_file(filepath, content):
    """파일 생성"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ {filepath}")

def main():
    print("="*70)
    print("📝 필수 파일 생성 중...")
    print("="*70)
    
    # 1. preprocessing/recipe_preprocessor.py
    create_file('preprocessing/recipe_preprocessor.py', '''import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class RecipePreprocessor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._rename_columns()
        
    def _rename_columns(self):
        column_mapping = {
            'RCP_SNO': 'recipe_id', 'RCP_TTL': 'title', 'CKG_NM': 'recipe_name',
            'RGTR_ID': 'user_id', 'RGTR_NM': 'user_name',
            'INQ_CNT': 'views', 'RCMM_CNT': 'recommendations', 'SRAP_CNT': 'scraps',
            'CKG_MTH_ACTO_NM': 'cooking_method', 'CKG_STA_ACTO_NM': 'situation',
            'CKG_MTRL_ACTO_NM': 'ingredients', 'CKG_KND_ACTO_NM': 'category',
            'CKG_IPDC': 'description', 'CKG_MTRL_CN': 'ingredients_detail',
            'CKG_INBUN_NM': 'servings', 'CKG_DODF_NM': 'difficulty',
            'CKG_TIME_NM': 'cooking_time', 'FIRST_REG_DT': 'registered_date',
            'RCP_IMG_URL': 'image_url'
        }
        self.df = self.df.rename(columns=column_mapping)
        print(f"✅ 컬럼명 변경 완료")
    
    def clean_data(self):
        print(f"📊 원본 데이터: {len(self.df):,}개")
        self.df = self.df.drop_duplicates(subset=['recipe_id'])
        self.df = self.df.dropna(subset=['recipe_id', 'title', 'category'])
        
        self.df['category'] = self.df['category'].fillna('기타')
        self.df['difficulty'] = self.df['difficulty'].fillna('중급')
        
        numeric_cols = ['views', 'recommendations', 'scraps']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
        
        self.recipe_id_mapping = {old_id: new_id for new_id, old_id in enumerate(self.df['recipe_id'].unique())}
        self.df['original_recipe_id'] = self.df['recipe_id']
        self.df['recipe_id'] = self.df['recipe_id'].map(self.recipe_id_mapping)
        
        print(f"✅ 데이터 정제 완료: {len(self.df):,}개")
        return self
    
    def extract_features(self):
        print("🔍 특징 추출 중...")
        
        def extract_time_minutes(time_str):
            if pd.isna(time_str): return 30
            time_str = str(time_str)
            if '분' in time_str:
                nums = re.findall(r'(\\d+)', time_str)
                return int(nums[0]) if nums else 30
            return 30
        
        self.df['cooking_time_minutes'] = self.df['cooking_time'].apply(extract_time_minutes)
        self.df['difficulty_level'] = self.df['difficulty'].map({'초급': 1, '중급': 2, '고급': 3}).fillna(2)
        
        max_views = self.df['views'].max() or 1
        max_rcmm = self.df['recommendations'].max() or 1
        max_scrap = self.df['scraps'].max() or 1
        
        self.df['popularity_score'] = (
            0.5 * (self.df['views'] / max_views) + 
            0.3 * (self.df['recommendations'] / max_rcmm) +
            0.2 * (self.df['scraps'] / max_scrap)
        )
        
        print(f"✅ 특징 추출 완료")
        return self
    
    def encode_categorical(self):
        print("🔢 카테고리 인코딩 중...")
        self.le_category = LabelEncoder()
        self.df['category_encoded'] = self.le_category.fit_transform(self.df['category'])
        print(f"✅ 인코딩 완료")
        return self
    
    def save(self, output_path):
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 저장: {output_path}")
        return self.df

if __name__ == "__main__":
    preprocessor = RecipePreprocessor('data/raw/recipes.csv')
    preprocessor.clean_data().extract_features().encode_categorical().save('data/processed/recipes_processed.csv')
    print("\\n✅ 전처리 완료!")
''')

    # 2. preprocessing/interaction_simulator.py
    create_file('preprocessing/interaction_simulator.py', '''import pandas as pd
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
''')

    print("\\n" + "="*70)
    print("✅ 기본 파일 생성 완료!")
    print("="*70)
    print("\\n다음 단계:")
    print("  1. python create_dummy_data.py")
    print("  2. python preprocessing/recipe_preprocessor.py")
    print("  3. python preprocessing/interaction_simulator.py")
    print("\\n⚠️  models/ncf.py와 training/train_ncf.py는 너무 길어서")
    print("   별도로 생성해야 합니다.")

if __name__ == "__main__":
    main()