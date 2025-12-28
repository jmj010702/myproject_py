import pandas as pd

df = pd.read_csv('data/processed/recipes_processed.csv')
print(f"총 레시피: {len(df)}개")
print("\n레시피 샘플 10개:")
print(df[['title', 'category', 'difficulty']].head(10))