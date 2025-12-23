// ============================================================================
// 📄 파일: RecipeRecommendation.java
// 위치: src/main/java/com/Recipe/recommandation/dto/RecipeRecommendation.java
// ============================================================================

package com.Recipe.recommandation.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class RecipeRecommendation {
    @JsonProperty("recipe_id")
    private Long recipeId;

    @JsonProperty("original_recipe_id")
    private Long originalRecipeId;

    private String title;
    private String category;
    private String difficulty;

    @JsonProperty("cooking_time")
    private String cookingTime;

    @JsonProperty("image_url")
    private String imageUrl;

    private Double score;

    @JsonProperty("popularity_score")
    private Double popularityScore;
}