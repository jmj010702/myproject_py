import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

class NCF(Model):
    """
    Neural Collaborative Filtering (NCF) 모델
    
    논문: "Neural Collaborative Filtering" (He et al., WWW 2017)
    
    아키텍처:
    - GMF (Generalized Matrix Factorization): Element-wise 곱
    - MLP (Multi-Layer Perceptron): 비선형 변환
    - NeuMF: GMF + MLP 결합
    """
    
    def __init__(self, num_users, num_recipes, 
                 embedding_dim=64, 
                 mlp_layers=[128, 64, 32, 16],
                 model_type='NeuMF'):
        """
        Args:
            num_users: 사용자 수
            num_recipes: 레시피 수
            embedding_dim: 임베딩 차원
            mlp_layers: MLP 레이어 크기 리스트
            model_type: 'GMF', 'MLP', 'NeuMF' 중 선택
        """
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_recipes = num_recipes
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        
        # GMF 부분
        self.gmf_user_embedding = layers.Embedding(
            num_users, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='gmf_user_embedding'
        )
        self.gmf_recipe_embedding = layers.Embedding(
            num_recipes, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='gmf_recipe_embedding'
        )
        
        # MLP 부분
        self.mlp_user_embedding = layers.Embedding(
            num_users, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='mlp_user_embedding'
        )
        self.mlp_recipe_embedding = layers.Embedding(
            num_recipes, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='mlp_recipe_embedding'
        )
        
        # MLP 레이어들
        self.mlp_layers = []
        for units in mlp_layers:
            self.mlp_layers.append(layers.Dense(
                units, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(1e-6)
            ))
            self.mlp_layers.append(layers.BatchNormalization())
            self.mlp_layers.append(layers.Dropout(0.2))
        
        # 최종 출력 레이어
        if model_type == 'NeuMF':
            # GMF + MLP 결합
            self.output_layer = layers.Dense(1, activation='sigmoid', name='output')
        else:
            self.output_layer = layers.Dense(1, activation='sigmoid', name='output')
    
    def call(self, inputs, training=False):
        """
        Forward pass
        
        Args:
            inputs: [user_ids, recipe_ids] 튜플
            training: 학습 모드 여부
        
        Returns:
            예측 점수 (0~1)
        """
        user_ids, recipe_ids = inputs
        
        if self.model_type == 'GMF':
            # GMF only
            gmf_user_vec = self.gmf_user_embedding(user_ids)
            gmf_recipe_vec = self.gmf_recipe_embedding(recipe_ids)
            gmf_output = layers.Multiply()([gmf_user_vec, gmf_recipe_vec])
            output = self.output_layer(gmf_output)
            
        elif self.model_type == 'MLP':
            # MLP only
            mlp_user_vec = self.mlp_user_embedding(user_ids)
            mlp_recipe_vec = self.mlp_recipe_embedding(recipe_ids)
            mlp_concat = layers.Concatenate()([mlp_user_vec, mlp_recipe_vec])
            
            x = mlp_concat
            for layer in self.mlp_layers:
                x = layer(x, training=training) if isinstance(layer, layers.Dropout) else layer(x)
            
            output = self.output_layer(x)
            
        else:  # NeuMF
            # GMF 경로
            gmf_user_vec = self.gmf_user_embedding(user_ids)
            gmf_recipe_vec = self.gmf_recipe_embedding(recipe_ids)
            gmf_output = layers.Multiply()([gmf_user_vec, gmf_recipe_vec])
            
            # MLP 경로
            mlp_user_vec = self.mlp_user_embedding(user_ids)
            mlp_recipe_vec = self.mlp_recipe_embedding(recipe_ids)
            mlp_concat = layers.Concatenate()([mlp_user_vec, mlp_recipe_vec])
            
            x = mlp_concat
            for layer in self.mlp_layers:
                x = layer(x, training=training) if isinstance(layer, layers.Dropout) else layer(x)
            
            # GMF + MLP 결합
            neumf_concat = layers.Concatenate()([gmf_output, x])
            output = self.output_layer(neumf_concat)
        
        return output
    
    def get_recipe_embeddings(self):
        """
        실시간 추천을 위한 레시피 임베딩 추출
        
        Returns:
            (num_recipes, embedding_dim) 형태의 numpy array
        """
        recipe_ids = np.arange(self.num_recipes)
        
        if self.model_type == 'GMF':
            embeddings = self.gmf_recipe_embedding(recipe_ids).numpy()
        elif self.model_type == 'MLP':
            embeddings = self.mlp_recipe_embedding(recipe_ids).numpy()
        else:  # NeuMF - GMF와 MLP 임베딩 결합
            gmf_emb = self.gmf_recipe_embedding(recipe_ids).numpy()
            mlp_emb = self.mlp_recipe_embedding(recipe_ids).numpy()
            embeddings = np.concatenate([gmf_emb, mlp_emb], axis=1)
        
        return embeddings
    
    def predict_for_user(self, user_id, candidate_recipe_ids, top_k=10):
        """
        특정 사용자에 대한 Top-K 추천
        
        Args:
            user_id: 사용자 ID (정수)
            candidate_recipe_ids: 추천 후보 레시피 ID 리스트
            top_k: 추천할 개수
        
        Returns:
            (recipe_ids, scores) 튜플
        """
        user_ids = np.full(len(candidate_recipe_ids), user_id)
        recipe_ids = np.array(candidate_recipe_ids)
        
        predictions = self.predict([user_ids, recipe_ids], verbose=0)
        predictions = predictions.flatten()
        
        # Top-K 선택
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_recipe_ids = recipe_ids[top_indices]
        top_scores = predictions[top_indices]
        
        return top_recipe_ids, top_scores


def build_ncf_model(num_users, num_recipes, model_type='NeuMF', 
                    embedding_dim=64, mlp_layers=[128, 64, 32, 16]):
    """
    NCF 모델 빌드 및 컴파일
    
    Returns:
        컴파일된 NCF 모델
    """
    model = NCF(
        num_users=num_users,
        num_recipes=num_recipes,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
        model_type=model_type
    )
    
    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


# 학습 데이터 생성 함수
class NCFDataGenerator:
    """NCF 학습을 위한 데이터 생성기"""
    
    def __init__(self, interactions_df, num_negatives=4):
        """
        Args:
            interactions_df: 상호작용 데이터프레임
            num_negatives: positive 샘플당 negative 샘플 수
        """
        self.interactions_df = interactions_df
        self.num_negatives = num_negatives
        
        # 사용자-레시피 상호작용 집합
        self.user_recipe_set = set(
            zip(interactions_df['user_id'], interactions_df['recipe_id'])
        )
        
        self.all_recipe_ids = interactions_df['recipe_id'].unique()
        self.user_ids = interactions_df['user_id'].unique()
    
    def generate_training_data(self):
        """
        학습 데이터 생성 (Negative Sampling)
        
        Returns:
            (user_ids, recipe_ids, labels) 튜플
        """
        user_list = []
        recipe_list = []
        label_list = []
        
        # Positive 샘플
        for _, row in self.interactions_df.iterrows():
            user_list.append(row['user_id'])
            recipe_list.append(row['recipe_id'])
            label_list.append(1)  # Positive
            
            # Negative 샘플 생성
            for _ in range(self.num_negatives):
                # 사용자가 상호작용하지 않은 레시피 랜덤 선택
                neg_recipe = self._sample_negative(row['user_id'])
                user_list.append(row['user_id'])
                recipe_list.append(neg_recipe)
                label_list.append(0)  # Negative
        
        return (
            np.array(user_list),
            np.array(recipe_list),
            np.array(label_list, dtype=np.float32)
        )
    
    def _sample_negative(self, user_id):
        """사용자가 상호작용하지 않은 레시피 샘플링"""
        while True:
            neg_recipe = np.random.choice(self.all_recipe_ids)
            if (user_id, neg_recipe) not in self.user_recipe_set:
                return neg_recipe


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = build_ncf_model(
        num_users=5000,
        num_recipes=20000,
        model_type='NeuMF',
        embedding_dim=64
    )
    
    # 모델 구조 출력
    print(model.summary())
    
    # 더미 데이터로 테스트
    user_ids = np.array([1, 2, 3, 4, 5])
    recipe_ids = np.array([100, 200, 300, 400, 500])
    
    predictions = model.predict([user_ids, recipe_ids])
    print(f"\n예측 결과: {predictions.flatten()}")