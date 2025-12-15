// ============================================================================
// 📄 파일 1: RecipeRecommendation.java
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


// ============================================================================
// 📄 파일 2: RecommendationResponse.java
// 위치: src/main/java/com/Recipe/recommandation/dto/RecommendationResponse.java
// ============================================================================

package com.Recipe.recommandation.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class RecommendationResponse {
    @JsonProperty("user_id")
    private Long userId;

    @JsonProperty("recipe_id")
    private Long recipeId;

    private List<RecipeRecommendation> recommendations;

    @JsonProperty("similar_recipes")
    private List<RecipeRecommendation> similarRecipes;

    private Integer count;
}