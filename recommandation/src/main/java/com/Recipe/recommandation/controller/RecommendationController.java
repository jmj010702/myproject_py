// ============================================================================
// 📄 파일: RecommendationController.java
// 위치: src/main/java/com/Recipe/recommandation/controller/RecommendationController.java
// ============================================================================

package com.Recipe.recommandation.controller;

import com.Recipe.recommandation.dto.RecommendationResponse;
import com.Recipe.recommandation.service.RecommendationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * 레시피 추천 API 컨트롤러
 */
@Slf4j
@RestController
@RequestMapping("/api/recommendations")
@RequiredArgsConstructor
public class RecommendationController {

    private final RecommendationService recommendationService;

    /**
     * 홈 화면 개인화 추천
     *
     * GET /api/recommendations/home?userId=123&topK=10
     */
    @GetMapping("/home")
    public ResponseEntity<RecommendationResponse> getHomeRecommendations(
            @RequestParam Long userId,
            @RequestParam(defaultValue = "10") Integer topK) {

        log.info("홈 추천 요청 - userId: {}, topK: {}", userId, topK);

        try {
            RecommendationResponse response = recommendationService.getPersonalizedRecommendations(
                    userId, topK, true
            );

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("홈 추천 실패", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * 유사 레시피 추천
     *
     * GET /api/recommendations/similar?recipeId=456&topK=5
     */
    @GetMapping("/similar")
    public ResponseEntity<RecommendationResponse> getSimilarRecipes(
            @RequestParam Long recipeId,
            @RequestParam(defaultValue = "5") Integer topK) {

        log.info("유사 레시피 요청 - recipeId: {}, topK: {}", recipeId, topK);

        try {
            RecommendationResponse response = recommendationService.getSimilarRecipes(
                    recipeId, topK
            );

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("유사 레시피 추천 실패", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * 사용자 상호작용 기록
     *
     * POST /api/recommendations/interaction
     */
    @PostMapping("/interaction")
    public ResponseEntity<Void> recordInteraction(
            @RequestParam Long userId,
            @RequestParam Long recipeId,
            @RequestParam(defaultValue = "view") String type) {

        log.debug("상호작용 기록 - userId: {}, recipeId: {}, type: {}",
                userId, recipeId, type);

        recommendationService.sendFeedback(userId, recipeId, type);

        return ResponseEntity.ok().build();
    }

    /**
     * 헬스 체크
     */
    @GetMapping("/health")
    public ResponseEntity<String> healthCheck() {
        return ResponseEntity.ok("Spring Boot Recommendation Service is running!");
    }
}