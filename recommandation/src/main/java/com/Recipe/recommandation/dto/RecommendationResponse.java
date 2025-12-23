// ============================================================================
// 📄 파일: RecommendationResponse.java
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