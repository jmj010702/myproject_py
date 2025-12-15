import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import sys

# NCF 모델 임포트 (위에서 작성한 ncf.py)
from models.ncf import NCF, NCFDataGenerator, build_ncf_model

class NCFTrainer:
    """NCF 모델 학습 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def load_data(self):
        """데이터 로드"""
        print("📂 데이터 로딩 중...")
        
        # 레시피 데이터
        self.recipes_df = pd.read_csv(self.config['recipes_path'])
        
        # 상호작용 데이터
        self.train_df = pd.read_csv(self.config['train_path'])
        self.test_df = pd.read_csv(self.config['test_path'])
        
        # 사용자 및 레시피 ID 범위 확인
        self.num_users = max(
            self.train_df['user_id'].max(),
            self.test_df['user_id'].max()
        ) + 1
        
        self.num_recipes = len(self.recipes_df)
        
        print(f"✅ 데이터 로드 완료")
        print(f"  - 사용자 수: {self.num_users:,}")
        print(f"  - 레시피 수: {self.num_recipes:,}")
        print(f"  - Train 상호작용: {len(self.train_df):,}")
        print(f"  - Test 상호작용: {len(self.test_df):,}")
        
    def prepare_training_data(self):
        """학습 데이터 준비 (Negative Sampling)"""
        print("\n🎲 학습 데이터 생성 중 (Negative Sampling)...")
        
        # Train 데이터 생성
        train_generator = NCFDataGenerator(
            self.train_df, 
            num_negatives=self.config['num_negatives']
        )
        self.train_user_ids, self.train_recipe_ids, self.train_labels = \
            train_generator.generate_training_data()
        
        # Test 데이터 생성
        test_generator = NCFDataGenerator(
            self.test_df,
            num_negatives=self.config['num_negatives']
        )
        self.test_user_ids, self.test_recipe_ids, self.test_labels = \
            test_generator.generate_training_data()
        
        print(f"✅ 학습 데이터 생성 완료")
        print(f"  - Train 샘플: {len(self.train_labels):,}개")
        print(f"  - Test 샘플: {len(self.test_labels):,}개")
        print(f"  - Positive 비율: {self.train_labels.mean()*100:.1f}%")
        
    def build_model(self):
        """모델 생성"""
        print(f"\n🏗️  NCF 모델 생성 중...")
        print(f"  - 모델 타입: {self.config['model_type']}")
        print(f"  - 임베딩 차원: {self.config['embedding_dim']}")
        print(f"  - MLP 레이어: {self.config['mlp_layers']}")
        
        self.model = build_ncf_model(
            num_users=self.num_users,
            num_recipes=self.num_recipes,
            model_type=self.config['model_type'],
            embedding_dim=self.config['embedding_dim'],
            mlp_layers=self.config['mlp_layers']
        )
        
        print("✅ 모델 생성 완료")
        
    def train(self):
        """모델 학습"""
        print(f"\n🚀 모델 학습 시작...")
        
        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.config['model_save_path'],
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 학습
        self.history = self.model.fit(
            [self.train_user_ids, self.train_recipe_ids],
            self.train_labels,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(
                [self.test_user_ids, self.test_recipe_ids],
                self.test_labels
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ 학습 완료!")
        
    def evaluate(self):
        """모델 평가"""
        print("\n📊 모델 평가 중...")
        
        results = self.model.evaluate(
            [self.test_user_ids, self.test_recipe_ids],
            self.test_labels,
            batch_size=self.config['batch_size'],
            verbose=0
        )
        
        print("\n최종 성능:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"  - {metric_name}: {value:.4f}")
        
        return dict(zip(self.model.metrics_names, results))
    
    def save_embeddings(self):
        """레시피 임베딩 저장 (실시간 추천용)"""
        print("\n💾 레시피 임베딩 저장 중...")
        
        recipe_embeddings = self.model.get_recipe_embeddings()
        
        embedding_path = self.config['embedding_save_path']
        np.save(embedding_path, recipe_embeddings)
        
        print(f"✅ 임베딩 저장 완료: {embedding_path}")
        print(f"  - Shape: {recipe_embeddings.shape}")
        
    def save_training_history(self):
        """학습 이력 저장"""
        history_path = self.config['model_save_path'].replace('.h5', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        print(f"📁 학습 이력 저장: {history_path}")


# 설정
CONFIG = {
    # 데이터 경로
    'recipes_path': 'data/processed/recipes_processed.csv',
    'train_path': 'data/processed/interactions_train.csv',
    'test_path': 'data/processed/interactions_test.csv',
    
    # 모델 하이퍼파라미터
    'model_type': 'NeuMF',  # 'GMF', 'MLP', 'NeuMF'
    'embedding_dim': 64,
    'mlp_layers': [128, 64, 32, 16],
    'num_negatives': 4,  # Positive 샘플당 Negative 샘플 수
    
    # 학습 설정
    'batch_size': 256,
    'epochs': 50,
    'learning_rate': 0.001,
    
    # 저장 경로
    'model_save_path': 'data/models/ncf_model.h5',
    'embedding_save_path': 'data/models/recipe_embeddings.npy'
}


def main():
    """메인 실행 함수"""
    print("="*70)
    print("🎯 NCF (Neural Collaborative Filtering) 모델 학습")
    print("="*70)
    
    # 디렉토리 생성
    os.makedirs('data/models', exist_ok=True)
    
    # 학습 시작
    trainer = NCFTrainer(CONFIG)
    
    # 1. 데이터 로드
    trainer.load_data()
    
    # 2. 학습 데이터 생성
    trainer.prepare_training_data()
    
    # 3. 모델 빌드
    trainer.build_model()
    
    # 4. 학습
    trainer.train()
    
    # 5. 평가
    results = trainer.evaluate()
    
    # 6. 임베딩 저장
    trainer.save_embeddings()
    
    # 7. 학습 이력 저장
    trainer.save_training_history()
    
    print("\n" + "="*70)
    print("✅ 모든 과정 완료!")
    print("="*70)
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()