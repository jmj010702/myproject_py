import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class RecipePreprocessor:
    def __init__(self, csv_path):
        """
        레시피 CSV 파일 전처리
        
        CSV 컬럼:
        RCP_SNO, RCP_TTL, CKG_NM, RGTR_ID, RGTR_NM, INQ_CNT, RCMM_CNT, SRAP_CNT,
        CKG_MTH_ACTO_NM, CKG_STA_ACTO_NM, CKG_MTRL_ACTO_NM, CKG_KND_ACTO_NM,
        CKG_IPDC, CKG_MTRL_CN, CKG_INBUN_NM, CKG_DODF_NM, CKG_TIME_NM,
        FIRST_REG_DT, RCP_IMG_URL
        """
        self.df = pd.read_csv(csv_path)
        self._rename_columns()
        
    def _rename_columns(self):
        """컬럼명을 영문으로 변경 (이해하기 쉽게)"""
        column_mapping = {
            'RCP_SNO': 'recipe_id',
            'RCP_TTL': 'title',
            'CKG_NM': 'recipe_name',
            'RGTR_ID': 'user_id',
            'RGTR_NM': 'user_name',
            'INQ_CNT': 'views',
            'RCMM_CNT': 'recommendations',
            'SRAP_CNT': 'scraps',
            'CKG_MTH_ACTO_NM': 'cooking_method',
            'CKG_STA_ACTO_NM': 'situation',
            'CKG_MTRL_ACTO_NM': 'ingredients',
            'CKG_KND_ACTO_NM': 'category',
            'CKG_IPDC': 'description',
            'CKG_MTRL_CN': 'ingredients_detail',
            'CKG_INBUN_NM': 'servings',
            'CKG_DODF_NM': 'difficulty',
            'CKG_TIME_NM': 'cooking_time',
            'FIRST_REG_DT': 'registered_date',
            'RCP_IMG_URL': 'image_url'
        }
        self.df = self.df.rename(columns=column_mapping)
        print(f"✅ 컬럼명 변경 완료")
    
    def clean_data(self):
        """데이터 정제"""
        print(f"📊 원본 데이터: {len(self.df):,}개")
        
        # 중복 제거
        before_dup = len(self.df)
        self.df = self.df.drop_duplicates(subset=['recipe_id'])
        print(f"  - 중복 제거: {before_dup - len(self.df):,}개")
        
        # 필수 컬럼 결측치 제거
        before_na = len(self.df)
        self.df = self.df.dropna(subset=['recipe_id', 'title', 'category'])
        print(f"  - 결측치 제거: {before_na - len(self.df):,}개")
        
        # 결측치 처리 (선택 컬럼)
        self.df['category'] = self.df['category'].fillna('기타')
        self.df['difficulty'] = self.df['difficulty'].fillna('중급')
        self.df['cooking_method'] = self.df['cooking_method'].fillna('기타')
        self.df['situation'] = self.df['situation'].fillna('일상')
        self.df['servings'] = self.df['servings'].fillna('2인분')
        self.df['cooking_time'] = self.df['cooking_time'].fillna('30분')
        
        # 숫자형 컬럼 변환
        numeric_cols = ['views', 'recommendations', 'scraps']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
        
        # recipe_id를 0부터 시작하는 연속된 정수로 매핑
        self.recipe_id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(self.df['recipe_id'].unique())
        }
        self.df['original_recipe_id'] = self.df['recipe_id']
        self.df['recipe_id'] = self.df['recipe_id'].map(self.recipe_id_mapping)
        
        print(f"✅ 데이터 정제 완료: {len(self.df):,}개 레시피")
        return self
    
    def extract_features(self):
        """특징 추출"""
        print("🔍 특징 추출 중...")
        
        # 1. 주재료 추출 (첫 번째 재료)
        def extract_main_ingredient(ingredients):
            if pd.isna(ingredients) or ingredients == '':
                return '기타'
            # 쉼표, 파이프(|), 슬래시로 구분
            parts = re.split(r'[,|/]', str(ingredients))
            if len(parts) > 0:
                return parts[0].strip()
            return '기타'
        
        self.df['main_ingredient'] = self.df['ingredients'].apply(extract_main_ingredient)
        
        # 2. 조리 시간 정규화 (분 단위)
        def extract_time_minutes(time_str):
            if pd.isna(time_str):
                return 30
            
            time_str = str(time_str).lower()
            
            # "2시간 이내", "1시간이내" 등
            if '시간' in time_str:
                hours = re.findall(r'(\d+)\s*시간', time_str)
                minutes = re.findall(r'(\d+)\s*분', time_str)
                total = 0
                if hours:
                    total += int(hours[0]) * 60
                if minutes:
                    total += int(minutes[0])
                return total if total > 0 else 60
            
            # "30분", "60분 이내" 등
            elif '분' in time_str:
                minutes = re.findall(r'(\d+)', time_str)
                return int(minutes[0]) if minutes else 30
            
            # 숫자만 있는 경우 (분으로 간주)
            elif time_str.isdigit():
                return int(time_str)
            
            return 30  # 기본값
        
        self.df['cooking_time_minutes'] = self.df['cooking_time'].apply(extract_time_minutes)
        
        # 3. 난이도 정규화
        difficulty_mapping = {
            '초급': 1, '아무나': 1, '쉬움': 1,
            '중급': 2, '보통': 2,
            '고급': 3, '어려움': 3
        }
        self.df['difficulty_level'] = self.df['difficulty'].map(
            lambda x: difficulty_mapping.get(str(x).strip(), 2)
        )
        
        # 4. 인기도 점수 계산 (정규화된 가중치 합)
        # 조회수(50%) + 추천수(30%) + 스크랩수(20%)
        max_views = self.df['views'].max() if self.df['views'].max() > 0 else 1
        max_rcmm = self.df['recommendations'].max() if self.df['recommendations'].max() > 0 else 1
        max_scrap = self.df['scraps'].max() if self.df['scraps'].max() > 0 else 1
        
        self.df['popularity_score'] = (
            0.5 * (self.df['views'] / max_views) + 
            0.3 * (self.df['recommendations'] / max_rcmm) +
            0.2 * (self.df['scraps'] / max_scrap)
        )
        
        # 5. 상황(occasion) 태그 추출
        self.df['occasion_tags'] = self.df['situation'].fillna('일상')
        
        print(f"✅ 특징 추출 완료")
        print(f"  - 주재료 종류: {self.df['main_ingredient'].nunique()}개")
        print(f"  - 평균 조리시간: {self.df['cooking_time_minutes'].mean():.1f}분")
        print(f"  - 난이도 분포: {dict(self.df['difficulty'].value_counts())}")
        
        return self
    
    def encode_categorical(self):
        """카테고리 변수 인코딩"""
        print("🔢 카테고리 인코딩 중...")
        
        # LabelEncoder 생성
        self.le_category = LabelEncoder()
        self.le_difficulty = LabelEncoder()
        self.le_method = LabelEncoder()
        self.le_ingredient = LabelEncoder()
        self.le_situation = LabelEncoder()
        
        # 인코딩
        self.df['category_encoded'] = self.le_category.fit_transform(self.df['category'])
        self.df['difficulty_encoded'] = self.le_difficulty.fit_transform(self.df['difficulty'])
        self.df['method_encoded'] = self.le_method.fit_transform(self.df['cooking_method'])
        self.df['ingredient_encoded'] = self.le_ingredient.fit_transform(self.df['main_ingredient'])
        self.df['situation_encoded'] = self.le_situation.fit_transform(self.df['situation'])
        
        print(f"✅ 인코딩 완료")
        print(f"  - 카테고리: {len(self.le_category.classes_)}개")
        print(f"  - 조리방법: {len(self.le_method.classes_)}개")
        print(f"  - 상황: {len(self.le_situation.classes_)}개")
        
        return self
    
    def create_content_features(self):
        """Content-Based 추천을 위한 텍스트 특징 생성"""
        # 재료, 카테고리, 조리방법, 상황을 하나의 텍스트로 결합
        self.df['content_text'] = (
            self.df['ingredients'].fillna('') + ' ' +
            self.df['category'].fillna('') + ' ' +
            self.df['cooking_method'].fillna('') + ' ' +
            self.df['situation'].fillna('') + ' ' +
            self.df['main_ingredient'].fillna('')
        )
        
        return self
    
    def get_statistics(self):
        """데이터 통계 출력"""
        print("\n" + "="*60)
        print("📊 최종 데이터 통계")
        print("="*60)
        print(f"총 레시피 수: {len(self.df):,}개")
        print(f"\n카테고리별 분포 (상위 10개):")
        print(self.df['category'].value_counts().head(10))
        print(f"\n난이도 분포:")
        print(self.df['difficulty'].value_counts())
        print(f"\n조리시간 통계:")
        print(f"  - 평균: {self.df['cooking_time_minutes'].mean():.1f}분")
        print(f"  - 중앙값: {self.df['cooking_time_minutes'].median():.1f}분")
        print(f"  - 최소: {self.df['cooking_time_minutes'].min():.0f}분")
        print(f"  - 최대: {self.df['cooking_time_minutes'].max():.0f}분")
        print(f"\n인기도 통계:")
        print(f"  - 평균 조회수: {self.df['views'].mean():.0f}")
        print(f"  - 평균 추천수: {self.df['recommendations'].mean():.0f}")
        print(f"  - 평균 스크랩수: {self.df['scraps'].mean():.0f}")
        print("="*60 + "\n")
    
    def save(self, output_path, save_mapping=True):
        """전처리된 데이터 저장"""
        # 메인 데이터 저장
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 전처리 완료: {len(self.df):,}개 레시피")
        print(f"📁 저장 위치: {output_path}")
        
        # recipe_id 매핑 저장 (역변환용)
        if save_mapping:
            mapping_df = pd.DataFrame([
                {'new_id': new_id, 'original_id': old_id}
                for old_id, new_id in self.recipe_id_mapping.items()
            ])
            mapping_path = output_path.replace('.csv', '_id_mapping.csv')
            mapping_df.to_csv(mapping_path, index=False)
            print(f"📁 ID 매핑 저장: {mapping_path}")
        
        # 통계 출력
        self.get_statistics()
        
        return self.df


# 실행 예시
if __name__ == "__main__":
    print("🚀 레시피 데이터 전처리 시작\n")
    
    preprocessor = RecipePreprocessor('data/raw/recipes.csv')
    df = (preprocessor
          .clean_data()
          .extract_features()
          .encode_categorical()
          .create_content_features()
          .save('data/processed/recipes_processed.csv'))
    
    print("\n✅ 전처리 완료!")
    print("\n샘플 데이터 (처음 3개):")
    print(df[['recipe_id', 'title', 'category', 'difficulty', 
              'cooking_time_minutes', 'popularity_score']].head(3))