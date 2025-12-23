from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow import keras
import json
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ncf import build_ncf_model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

class RecommendationSystem:
    def __init__(self):
        self.model = None
        self.recipes_df = None
        self.recipe_embeddings = None
        self.user_history = {}
        
    def load_models(self):
        print("🔄 모델 로딩 중...")
        
        # 레시피 데이터 로드
        self.recipes_df = pd.read_csv('data/processed/recipes_processed.csv')
        
        # 레시피 임베딩 로드
        self.recipe_embeddings = np.load('data/models/recipe_embeddings.npy')
        
        # 모델 구조 정보 로드
        with open('data/models/model_config.json', 'r') as f:
            model_config = json.load(f)
        
        # 모델 재생성
        self.model = build_ncf_model(
            num_users=model_config['num_users'],
            num_recipes=model_config['num_recipes'],
            embedding_dim=model_config['embedding_dim'],
            mlp_layers=model_config['mlp_layers']
        )
        
        # 모델 빌드 (더미 데이터로 한 번 호출)
        dummy_users = np.array([0])
        dummy_recipes = np.array([0])
        _ = self.model.predict([dummy_users, dummy_recipes], verbose=0)
        
        # 가중치 로드
        self.model.load_weights('data/models/ncf_model.weights.h5')
        
        print("✅ 모델 로드 완료")
        print(f"  - 레시피 수: {len(self.recipes_df):,}")
        print(f"  - 임베딩 Shape: {self.recipe_embeddings.shape}")
    
    def get_ncf_recommendations(self, user_id, exclude_recipe_ids=None, top_k=20):
        """NCF 기반 추천"""
        if exclude_recipe_ids is None:
            exclude_recipe_ids = set()
        
        all_recipe_ids = np.arange(len(self.recipes_df))
        user_ids = np.full(len(all_recipe_ids), user_id)
        
        # 배치 예측
        batch_size = 1024
        predictions = []
        
        for i in range(0, len(all_recipe_ids), batch_size):
            batch_users = user_ids[i:i+batch_size]
            batch_recipes = all_recipe_ids[i:i+batch_size]
            batch_preds = self.model.predict(
                [batch_users, batch_recipes], 
                verbose=0
            )
            predictions.extend(batch_preds.flatten())
        
        predictions = np.array(predictions)
        
        # 이미 본 레시피 제외
        for recipe_id in exclude_recipe_ids:
            if recipe_id < len(predictions):
                predictions[recipe_id] = -1
        
        # Top-K 선택
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_scores = predictions[top_indices]
        
        return top_indices, top_scores
    
    def get_content_based_recommendations(self, recipe_id, top_k=10):
        """Content-Based 추천 (유사 레시피)"""
        if recipe_id >= len(self.recipe_embeddings):
            return [], []
        
        target_embedding = self.recipe_embeddings[recipe_id].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, self.recipe_embeddings)[0]
        
        # 자기 자신 제외
        similarities[recipe_id] = -1
        
        # Top-K 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def diversify_recommendations(self, recipe_ids, scores, lambda_param=0.5):
        """다양성 보장 (MMR)"""
        if len(recipe_ids) == 0:
            return [], []
        
        selected_ids = []
        selected_scores = []
        remaining_ids = list(recipe_ids)
        remaining_scores = list(scores)
        
        # 첫 번째는 가장 높은 점수
        max_idx = np.argmax(remaining_scores)
        selected_ids.append(remaining_ids[max_idx])
        selected_scores.append(remaining_scores[max_idx])
        del remaining_ids[max_idx]
        del remaining_scores[max_idx]
        
        # MMR로 나머지 선택
        while remaining_ids and len(selected_ids) < len(recipe_ids):
            mmr_scores = []
            
            for i, (rid, score) in enumerate(zip(remaining_ids, remaining_scores)):
                relevance = score
                
                if rid < len(self.recipe_embeddings):
                    candidate_emb = self.recipe_embeddings[rid]
                    max_sim = 0
                    
                    for selected_id in selected_ids:
                        if selected_id < len(self.recipe_embeddings):
                            selected_emb = self.recipe_embeddings[selected_id]
                            sim = cosine_similarity(
                                candidate_emb.reshape(1, -1),
                                selected_emb.reshape(1, -1)
                            )[0][0]
                            max_sim = max(max_sim, sim)
                    
                    diversity = 1 - max_sim
                else:
                    diversity = 0.5
                
                mmr = lambda_param * relevance + (1 - lambda_param) * diversity
                mmr_scores.append(mmr)
            
            best_idx = np.argmax(mmr_scores)
            selected_ids.append(remaining_ids[best_idx])
            selected_scores.append(remaining_scores[best_idx])
            del remaining_ids[best_idx]
            del remaining_scores[best_idx]
        
        return selected_ids, selected_scores
    
    def hybrid_recommendations(self, user_id, top_k=10, diversity=True):
        """하이브리드 추천"""
        user_history = self.user_history.get(user_id, set())
        
        # NCF 추천
        ncf_ids, ncf_scores = self.get_ncf_recommendations(
            user_id, 
            exclude_recipe_ids=user_history,
            top_k=top_k * 2
        )
        
        # 점수 정규화
        if len(ncf_scores) > 0:
            ncf_scores = (ncf_scores - ncf_scores.min()) / (ncf_scores.max() - ncf_scores.min() + 1e-8)
        
        # 다양성 적용
        if diversity and len(ncf_ids) > 0:
            final_ids, final_scores = self.diversify_recommendations(
                ncf_ids, ncf_scores, lambda_param=0.6
            )
        else:
            final_ids, final_scores = ncf_ids, ncf_scores
        
        # Top-K만 선택
        final_ids = final_ids[:top_k]
        final_scores = final_scores[:top_k]
        
        return final_ids, final_scores
    
    def format_recommendations(self, recipe_ids, scores):
        """추천 결과 포맷팅"""
        recommendations = []
        
        for recipe_id, score in zip(recipe_ids, scores):
            if recipe_id < len(self.recipes_df):
                recipe = self.recipes_df.iloc[recipe_id]
                recommendations.append({
                    'recipe_id': int(recipe['recipe_id']),
                    'original_recipe_id': int(recipe.get('original_recipe_id', recipe['recipe_id'])),
                    'title': recipe['title'],
                    'category': recipe['category'],
                    'difficulty': recipe['difficulty'],
                    'cooking_time': str(recipe.get('cooking_time', '')),
                    'image_url': str(recipe.get('image_url', '')),
                    'score': float(score),
                    'popularity_score': float(recipe.get('popularity_score', 0))
                })
        
        return recommendations
    
    def update_user_history(self, user_id, recipe_id, interaction_type):
        """사용자 히스토리 업데이트"""
        if user_id not in self.user_history:
            self.user_history[user_id] = set()
        self.user_history[user_id].add(recipe_id)


# 전역 추천 시스템 인스턴스
rec_system = RecommendationSystem()


# API 엔드포인트
@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': rec_system.model is not None
    })


@app.route('/recommend/personalized', methods=['POST'])
def get_personalized_recommendations():
    """개인화 추천"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_k = data.get('top_k', 10)
        diversity = data.get('diversity', True)
        
        if user_id is None:
            return jsonify({'error': 'user_id is required'}), 400
        
        recipe_ids, scores = rec_system.hybrid_recommendations(
            user_id=user_id,
            top_k=top_k,
            diversity=diversity
        )
        
        recommendations = rec_system.format_recommendations(recipe_ids, scores)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recommend/similar', methods=['POST'])
def get_similar_recipes():
    """유사 레시피 추천"""
    try:
        data = request.get_json()
        recipe_id = data.get('recipe_id')
        top_k = data.get('top_k', 10)
        
        if recipe_id is None:
            return jsonify({'error': 'recipe_id is required'}), 400
        
        recipe_ids, scores = rec_system.get_content_based_recommendations(
            recipe_id=recipe_id,
            top_k=top_k
        )
        
        recommendations = rec_system.format_recommendations(recipe_ids, scores)
        
        return jsonify({
            'recipe_id': recipe_id,
            'similar_recipes': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """사용자 피드백 수집"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        recipe_id = data.get('recipe_id')
        interaction_type = data.get('interaction_type', 'view')
        
        if user_id is None or recipe_id is None:
            return jsonify({'error': 'user_id and recipe_id are required'}), 400
        
        rec_system.update_user_history(user_id, recipe_id, interaction_type)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback collected'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*70)
    print("🚀 Flask 추천 서버 시작")
    print("="*70)
    
    rec_system.load_models()
    
    print("\n📡 서버 실행 중...")
    print("  - URL: http://localhost:5000")
    print("  - 엔드포인트:")
    print("    • POST /recommend/personalized - 개인화 추천")
    print("    • POST /recommend/similar - 유사 레시피")
    print("    • POST /feedback - 피드백 수집")
    print("    • GET /health - 헬스 체크")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)