import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class InteractionSimulator:
    """
    더미 사용자 및 상호작용 데이터 생성
    - 현실적인 사용자 행동 패턴 시뮬레이션
    - NCF 학습에 필요한 implicit feedback 생성
    """
    
    def __init__(self, recipes_df, num_users=5000):
        self.recipes_df = recipes_df
        self.num_users = num_users
        self.num_recipes = len(recipes_df)
        
        # 사용자 페르소나 정의
        self.user_personas = {
            'health_conscious': 0.2,    # 건강식 선호
            'quick_cook': 0.3,          # 간편식 선호
            'gourmet': 0.15,            # 복잡한 요리 선호
            'traditional': 0.2,         # 전통 음식 선호
            'random': 0.15              # 무작위
        }
        
    def generate_users(self):
        """더미 사용자 생성"""
        users = []
        
        for user_id in range(1, self.num_users + 1):
            # 페르소나 할당
            persona = np.random.choice(
                list(self.user_personas.keys()),
                p=list(self.user_personas.values())
            )
            
            # 사용자 선호도 생성
            preferred_categories = self._get_preferred_categories(persona)
            preferred_difficulty = self._get_preferred_difficulty(persona)
            max_cooking_time = self._get_max_cooking_time(persona)
            
            users.append({
                'user_id': user_id,
                'persona': persona,
                'preferred_categories': ','.join(preferred_categories),
                'preferred_difficulty': preferred_difficulty,
                'max_cooking_time': max_cooking_time,
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            })
        
        self.users_df = pd.DataFrame(users)
        print(f"✅ {self.num_users}명의 더미 사용자 생성 완료")
        return self.users_df
    
    def _get_preferred_categories(self, persona):
        """페르소나별 선호 카테고리"""
        category_map = {
            'health_conscious': ['국/탕', '메인반찬', '샐러드'],
            'quick_cook': ['일품', '간식', '볶음'],
            'gourmet': ['메인반찬', '양식', '중식'],
            'traditional': ['국/탕', '김치/젓갈/장류', '메인반찬'],
            'random': list(self.recipes_df['category'].unique()[:3])
        }
        return category_map.get(persona, ['메인반찬'])
    
    def _get_preferred_difficulty(self, persona):
        """페르소나별 선호 난이도"""
        difficulty_map = {
            'health_conscious': '초급',
            'quick_cook': '초급',
            'gourmet': '고급',
            'traditional': '중급',
            'random': np.random.choice(['초급', '중급', '고급'])
        }
        return difficulty_map.get(persona, '중급')
    
    def _get_max_cooking_time(self, persona):
        """페르소나별 최대 조리 시간 (분)"""
        time_map = {
            'health_conscious': 60,
            'quick_cook': 30,
            'gourmet': 120,
            'traditional': 90,
            'random': 60
        }
        return time_map.get(persona, 60)
    
    def generate_interactions(self, interactions_per_user_range=(5, 50)):
        """사용자-레시피 상호작용 생성"""
        interactions = []
        
        for _, user in self.users_df.iterrows():
            # 활동량에 따른 상호작용 수
            activity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            num_interactions = int(
                np.random.randint(*interactions_per_user_range) * 
                activity_multiplier[user['activity_level']]
            )
            
            # 사용자 선호도에 맞는 레시피 필터링
            candidate_recipes = self._filter_recipes_by_preference(user)
            
            # 상호작용 생성
            for _ in range(num_interactions):
                recipe = self._select_recipe(candidate_recipes, user)
                
                # 상호작용 타입 및 implicit feedback 점수
                interaction_type, score = self._generate_interaction_type()
                
                # 타임스탬프 (최근 6개월 내)
                timestamp = self._generate_timestamp()
                
                interactions.append({
                    'user_id': user['user_id'],
                    'recipe_id': recipe['recipe_id'],
                    'interaction_type': interaction_type,
                    'implicit_score': score,
                    'timestamp': timestamp
                })
        
        self.interactions_df = pd.DataFrame(interactions)
        print(f"✅ {len(interactions):,}개의 상호작용 생성 완료")
        print(f"  - 평균 사용자당 상호작용: {len(interactions) / self.num_users:.1f}개")
        
        return self.interactions_df
    
    def _filter_recipes_by_preference(self, user):
        """사용자 선호도에 맞는 레시피 필터링"""
        preferred_cats = user['preferred_categories'].split(',')
        
        # 70% 확률로 선호 카테고리, 30% 확률로 랜덤
        if np.random.random() < 0.7:
            filtered = self.recipes_df[
                self.recipes_df['category'].isin(preferred_cats)
            ]
        else:
            filtered = self.recipes_df
        
        return filtered if len(filtered) > 0 else self.recipes_df
    
    def _select_recipe(self, candidate_recipes, user):
        """레시피 선택 (인기도 기반 확률적 선택)"""
        # 인기도에 따른 가중치
        weights = candidate_recipes['popularity_score'].values
        weights = weights / weights.sum()
        
        idx = np.random.choice(len(candidate_recipes), p=weights)
        return candidate_recipes.iloc[idx]
    
    def _generate_interaction_type(self):
        """상호작용 타입 및 점수 생성"""
        # view(1점) > click(2점) > like(3점)
        interaction_types = {
            'view': (1, 0.6),
            'click': (2, 0.3),
            'like': (3, 0.1)
        }
        
        interaction_type = np.random.choice(
            list(interaction_types.keys()),
            p=[v[1] for v in interaction_types.values()]
        )
        
        score = interaction_types[interaction_type][0]
        return interaction_type, score
    
    def _generate_timestamp(self):
        """최근 6개월 내 타임스탬프 생성"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        return random_date.strftime('%Y-%m-%d %H:%M:%S')
    
    def split_train_test(self, test_ratio=0.2):
        """Train/Test 분할 (시간 기준)"""
        # 타임스탬프로 정렬
        self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        self.interactions_df = self.interactions_df.sort_values('timestamp')
        
        # 각 사용자별로 최신 20%를 테스트로
        train_list = []
        test_list = []
        
        for user_id in self.interactions_df['user_id'].unique():
            user_data = self.interactions_df[self.interactions_df['user_id'] == user_id]
            split_idx = int(len(user_data) * (1 - test_ratio))
            
            train_list.append(user_data.iloc[:split_idx])
            test_list.append(user_data.iloc[split_idx:])
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        print(f"\n📊 데이터 분할:")
        print(f"  - Train: {len(train_df):,}개 ({len(train_df)/len(self.interactions_df)*100:.1f}%)")
        print(f"  - Test: {len(test_df):,}개 ({len(test_df)/len(self.interactions_df)*100:.1f}%)")
        
        return train_df, test_df
    
    def save(self, users_path, train_path, test_path):
        """데이터 저장"""
        self.users_df.to_csv(users_path, index=False)
        
        train_df, test_df = self.split_train_test()
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n✅ 데이터 저장 완료:")
        print(f"  - 사용자: {users_path}")
        print(f"  - Train: {train_path}")
        print(f"  - Test: {test_path}")

# 실행 예시
if __name__ == "__main__":
    # 전처리된 레시피 로드
    recipes_df = pd.read_csv('data/processed/recipes_processed.csv')
    
    # 시뮬레이터 실행
    simulator = InteractionSimulator(recipes_df, num_users=5000)
    simulator.generate_users()
    simulator.generate_interactions(interactions_per_user_range=(10, 50))
    simulator.save(
        'data/processed/users_dummy.csv',
        'data/processed/interactions_train.csv',
        'data/processed/interactions_test.csv'
    )