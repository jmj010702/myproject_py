import pandas as pd
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
                nums = re.findall(r'(\d+)', time_str)
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
    print("\n✅ 전처리 완료!")
