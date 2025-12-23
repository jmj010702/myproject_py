import pandas as pd
import numpy as np
from tensorflow import keras
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.ncf import NCF, NCFDataGenerator, build_ncf_model

class NCFTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def load_data(self):
        print("📂 데이터 로딩...")
        self.recipes_df = pd.read_csv(self.config['recipes_path'])
        self.train_df = pd.read_csv(self.config['train_path'])
        self.test_df = pd.read_csv(self.config['test_path'])
        self.num_users = max(self.train_df['user_id'].max(), self.test_df['user_id'].max()) + 1
        self.num_recipes = len(self.recipes_df)
        print(f"✅ 사용자: {self.num_users:,}, 레시피: {self.num_recipes:,}")
        print(f"   Train: {len(self.train_df):,}, Test: {len(self.test_df):,}")
        
    def prepare_training_data(self):
        print("\n🎲 Negative Sampling...")
        train_generator = NCFDataGenerator(self.train_df, num_negatives=self.config['num_negatives'])
        self.train_user_ids, self.train_recipe_ids, self.train_labels = train_generator.generate_training_data()
        test_generator = NCFDataGenerator(self.test_df, num_negatives=self.config['num_negatives'])
        self.test_user_ids, self.test_recipe_ids, self.test_labels = test_generator.generate_training_data()
        print(f"✅ Train: {len(self.train_labels):,}, Test: {len(self.test_labels):,}")
        
    def build_model(self):
        print(f"\n🏗️  NCF 모델 생성...")
        self.model = build_ncf_model(
            num_users=self.num_users,
            num_recipes=self.num_recipes,
            embedding_dim=self.config['embedding_dim'],
            mlp_layers=self.config['mlp_layers']
        )
        print("✅ 모델 생성 완료")
        
    def train(self):
        print(f"\n🚀 학습 시작 (Epochs: {self.config['epochs']})...")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            [self.train_user_ids, self.train_recipe_ids],
            self.train_labels,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=([self.test_user_ids, self.test_recipe_ids], self.test_labels),
            callbacks=callbacks,
            verbose=1
        )
        print("\n✅ 학습 완료!")
        
    def evaluate(self):
        print("\n📊 평가 중...")
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
    
    def save_models(self):
        print("\n💾 모델 가중치 및 임베딩 저장...")
        
        # 가중치만 저장 (전체 모델 대신)
        self.model.save_weights(self.config['weights_save_path'])
        print(f"✅ 가중치 저장: {self.config['weights_save_path']}")
        
        # 임베딩 저장
        recipe_embeddings = self.model.get_recipe_embeddings()
        np.save(self.config['embedding_save_path'], recipe_embeddings)
        print(f"✅ 임베딩 저장: {self.config['embedding_save_path']}")
        print(f"   Shape: {recipe_embeddings.shape}")
        
        # 모델 구조 정보 저장 (나중에 재구성용)
        import json
        model_config = {
            'num_users': self.num_users,
            'num_recipes': self.num_recipes,
            'embedding_dim': self.config['embedding_dim'],
            'mlp_layers': self.config['mlp_layers']
        }
        with open(self.config['config_save_path'], 'w') as f:
            json.dump(model_config, f)
        print(f"✅ 모델 구조 저장: {self.config['config_save_path']}")

CONFIG = {
    'recipes_path': 'data/processed/recipes_processed.csv',
    'train_path': 'data/processed/interactions_train.csv',
    'test_path': 'data/processed/interactions_test.csv',
    'embedding_dim': 64,
    'mlp_layers': [128, 64, 32, 16],
    'num_negatives': 4,
    'batch_size': 512,
    'epochs': 10,
    'weights_save_path': 'data/models/ncf_model.weights.h5',
    'embedding_save_path': 'data/models/recipe_embeddings.npy',
    'config_save_path': 'data/models/model_config.json'
}

def main():
    print("="*70)
    print("🎯 NCF 모델 학습")
    print("="*70)
    os.makedirs('data/models', exist_ok=True)
    trainer = NCFTrainer(CONFIG)
    trainer.load_data()
    trainer.prepare_training_data()
    trainer.build_model()
    trainer.train()
    trainer.evaluate()
    trainer.save_models()
    print("\n" + "="*70)
    print("✅ 모든 과정 완료!")
    print("="*70)
    return trainer

if __name__ == "__main__":
    trainer = main()