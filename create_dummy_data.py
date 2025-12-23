import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_dummy_recipes(num_recipes=1000):
    categories = ['국/탕', '메인반찬', '일품', '볶음', '찌개', '샐러드']
    difficulties = ['초급', '중급', '고급']
    cooking_times = ['10분', '20분', '30분', '60분']
    main_ingredients = ['돼지고기', '소고기', '닭고기', '생선', '두부', '계란']
    
    recipes = []
    for i in range(num_recipes):
        recipe_id = 1000000 + i
        ingredient = random.choice(main_ingredients)
        title = f"맛있는 {ingredient} 요리 {i+1}"
        
        recipes.append({
            'RCP_SNO': recipe_id, 'RCP_TTL': title, 'CKG_NM': title,
            'RGTR_ID': f'user_{random.randint(1, 100)}',
            'RGTR_NM': f'요리사{random.randint(1, 100)}',
            'INQ_CNT': random.randint(100, 50000),
            'RCMM_CNT': random.randint(10, 5000),
            'SRAP_CNT': random.randint(5, 1000),
            'CKG_MTH_ACTO_NM': random.choice(['굽기', '볶기', '찌기']),
            'CKG_STA_ACTO_NM': random.choice(['일상', '손님접대', '간편식']),
            'CKG_MTRL_ACTO_NM': f"{ingredient}, 양파, 마늘",
            'CKG_KND_ACTO_NM': random.choice(categories),
            'CKG_IPDC': f"{title} 레시피",
            'CKG_MTRL_CN': f"{ingredient} 300g",
            'CKG_INBUN_NM': random.choice(['2인분', '3인분', '4인분']),
            'CKG_DODF_NM': random.choice(difficulties),
            'CKG_TIME_NM': random.choice(cooking_times),
            'FIRST_REG_DT': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'RCP_IMG_URL': f'https://example.com/recipe_{recipe_id}.jpg'
        })
    
    return pd.DataFrame(recipes)

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    df = create_dummy_recipes(1000)
    df.to_csv('data/raw/recipes.csv', index=False, encoding='utf-8-sig')
    print(f"✅ {len(df)}개 레시피 생성: data/raw/recipes.csv")
