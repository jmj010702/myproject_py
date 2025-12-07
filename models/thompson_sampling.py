import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

class ThompsonSamplingEvaluator:
    """
    Thompson Sampling을 사용한 추천 시스템 평가
    
    Multi-Armed Bandit 문제로 추천 알고리즘을 평가:
    - 각 추천 알고리즘 = 하나의 Arm
    - 클릭/좋아요 = Reward
    """
    
    def __init__(self, algorithms):
        """
        Args:
            algorithms: 평가할 알고리즘 리스트
                        예: ['NCF', 'MF', 'Content-Based']
        """
        self.algorithms = algorithms
        self.n_algorithms = len(algorithms)
        
        # 각 알고리즘의 성공/실패 카운트 (Beta 분포 파라미터)
        self.alpha = {alg: 1 for alg in algorithms}  # 성공 횟수 + 1
        self.beta = {alg: 1 for alg in algorithms}   # 실패 횟수 + 1
        
        # 통계
        self.total_selections = {alg: 0 for alg in algorithms}
        self.total_rewards = {alg: 0 for alg in algorithms}
        self.cumulative_regret = []
        
    def select_algorithm(self):
        """
        Thompson Sampling으로 알고리즘 선택
        
        Returns:
            선택된 알고리즘 이름
        """
        samples = {}
        
        for alg in self.algorithms:
            # Beta 분포에서 샘플링
            samples[alg] = np.random.beta(self.alpha[alg], self.beta[alg])
        
        # 가장 높은 샘플 값을 가진 알고리즘 선택
        selected = max(samples, key=samples.get)
        return selected
    
    def update(self, algorithm, reward):
        """
        선택된 알고리즘의 성능 업데이트
        
        Args:
            algorithm: 선택된 알고리즘
            reward: 보상 (1=성공, 0=실패)
        """
        self.total_selections[algorithm] += 1
        
        if reward > 0:
            self.alpha[algorithm] += 1
            self.total_rewards[algorithm] += reward
        else:
            self.beta[algorithm] += 1
    
    def get_statistics(self):
        """현재 통계 반환"""
        stats = {}
        
        for alg in self.algorithms:
            total = self.total_selections[alg]
            if total > 0:
                ctr = self.total_rewards[alg] / total
                confidence = self.alpha[alg] / (self.alpha[alg] + self.beta[alg])
            else:
                ctr = 0
                confidence = 0.5
            
            stats[alg] = {
                'selections': total,
                'rewards': self.total_rewards[alg],
                'ctr': ctr,
                'confidence': confidence,
                'alpha': self.alpha[alg],
                'beta': self.beta[alg]
            }
        
        return stats
    
    def plot_results(self, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 알고리즘별 선택 횟수
        ax1 = axes[0, 0]
        selections = [self.total_selections[alg] for alg in self.algorithms]
        ax1.bar(self.algorithms, selections, color='skyblue')
        ax1.set_title('Algorithm Selection Count')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Selections')
        
        # 2. 알고리즘별 CTR (Click-Through Rate)
        ax2 = axes[0, 1]
        ctrs = []
        for alg in self.algorithms:
            total = self.total_selections[alg]
            ctr = self.total_rewards[alg] / total if total > 0 else 0
            ctrs.append(ctr * 100)
        ax2.bar(self.algorithms, ctrs, color='lightgreen')
        ax2.set_title('Click-Through Rate (CTR)')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('CTR (%)')
        
        # 3. Beta 분포 시각화
        ax3 = axes[1, 0]
        x = np.linspace(0, 1, 100)
        for alg in self.algorithms:
            from scipy.stats import beta as beta_dist
            y = beta_dist.pdf(x, self.alpha[alg], self.beta[alg])
            ax3.plot(x, y, label=alg, linewidth=2)
        ax3.set_title('Posterior Distributions (Beta)')
        ax3.set_xlabel('Success Probability')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 신뢰도 점수
        ax4 = axes[1, 1]
        confidences = []
        for alg in self.algorithms:
            conf = self.alpha[alg] / (self.alpha[alg] + self.beta[alg])
            confidences.append(conf * 100)
        ax4.bar(self.algorithms, confidences, color='coral')
        ax4.set_title('Confidence Score')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Confidence (%)')
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 그래프 저장: {save_path}")
        
        plt.show()


def simulate_online_evaluation(test_data, models_predictions, n_iterations=1000):
    """
    온라인 평가 시뮬레이션
    
    Args:
        test_data: 테스트 데이터프레임 (user_id, recipe_id, label)
        models_predictions: 각 모델의 예측 결과 딕셔너리
                           {'NCF': predictions, 'MF': predictions, ...}
        n_iterations: 시뮬레이션 반복 횟수
    """
    
    algorithms = list(models_predictions.keys())
    evaluator = ThompsonSamplingEvaluator(algorithms)
    
    print("🎰 Thompson Sampling 온라인 평가 시작\n")
    print(f"알고리즘: {algorithms}")
    print(f"반복 횟수: {n_iterations}\n")
    
    # 시뮬레이션
    for i in range(n_iterations):
        # 알고리즘 선택
        selected_alg = evaluator.select_algorithm()
        
        # 랜덤 사용자-레시피 쌍 선택
        idx = np.random.randint(len(test_data))
        true_label = test_data.iloc[idx]['implicit_score']  # 실제 레이블
        
        # 선택된 알고리즘의 예측
        prediction = models_predictions[selected_alg][idx]
        
        # 보상 계산 (예측이 맞으면 1, 틀리면 0)
        # 실제로는 사용자가 클릭했는지 여부
        reward = 1 if (prediction > 0.5 and true_label > 0) else 0
        
        # 업데이트
        evaluator.update(selected_alg, reward)
        
        # 중간 결과 출력
        if (i + 1) % 200 == 0:
            stats = evaluator.get_statistics()
            print(f"\n반복 {i+1}/{n_iterations}:")
            for alg, stat in stats.items():
                print(f"  {alg:15s}: "
                      f"선택 {stat['selections']:4d}회, "
                      f"CTR {stat['ctr']*100:5.2f}%, "
                      f"신뢰도 {stat['confidence']*100:5.2f}%")
    
    # 최종 결과
    print("\n" + "="*70)
    print("📊 최종 결과")
    print("="*70)
    
    stats = evaluator.get_statistics()
    results_df = pd.DataFrame(stats).T
    results_df = results_df.round(4)
    print(results_df)
    
    # 승자 결정
    best_alg = max(stats.keys(), key=lambda x: stats[x]['ctr'])
    print(f"\n🏆 최고 성능 알고리즘: {best_alg}")
    print(f"   CTR: {stats[best_alg]['ctr']*100:.2f}%")
    
    # 시각화
    evaluator.plot_results('thompson_sampling_results.png')
    
    return evaluator, stats


# 실행 예시
if __name__ == "__main__":
    # 더미 데이터로 테스트
    np.random.seed(42)
    
    # 테스트 데이터 생성
    n_samples = 1000
    test_data = pd.DataFrame({
        'user_id': np.random.randint(0, 100, n_samples),
        'recipe_id': np.random.randint(0, 500, n_samples),
        'implicit_score': np.random.randint(0, 2, n_samples)
    })
    
    # 각 모델의 예측 (더미)
    # 실제로는 학습된 모델의 예측값 사용
    models_predictions = {
        'NCF': np.random.rand(n_samples) * 0.7 + 0.15,  # 더 좋은 성능
        'MF': np.random.rand(n_samples) * 0.6 + 0.1,
        'Content-Based': np.random.rand(n_samples) * 0.5 + 0.05
    }
    
    # 평가 실행
    evaluator, stats = simulate_online_evaluation(
        test_data, 
        models_predictions, 
        n_iterations=1000
    )