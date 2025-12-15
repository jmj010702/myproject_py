// ============================================================================
// 📄 파일: RecommendationService.java
// 위치: src/main/java/com/Recipe/recommandation/service/RecommendationService.java
// ============================================================================

package com.Recipe.recommandation.service;

import com.Recipe.recommandation.dto.RecommendationResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class RecommendationService {

    private final RestTemplate restTemplate;

    @Value("${recommendation.api.base-url}")
    private String flaskBaseUrl;

    /**
     * 개인화 추천 가져오기
     */
    public RecommendationResponse getPersonalizedRecommendations(Long userId, Integer topK, Boolean diversity) {
        String url = flaskBaseUrl + "/recommend/personalized";

        try {
            // 요청 바디 생성
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("user_id", userId);
            requestBody.put("top_k", topK);
            requestBody.put("diversity", diversity);

            // 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

            // Flask API 호출
            ResponseEntity<RecommendationResponse> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    entity,
                    RecommendationResponse.class
            );

            log.info("개인화 추천 성공 - userId: {}, count: {}",
                    userId, response.getBody().getCount());

            return response.getBody();

        } catch (Exception e) {
            log.error("개인화 추천 실패 - userId: {}, error: {}", userId, e.getMessage());
            throw new RuntimeException("추천 시스템 오류", e);
        }
    }

    /**
     * 유사 레시피 추천 가져오기
     */
    public RecommendationResponse getSimilarRecipes(Long recipeId, Integer topK) {
        String url = flaskBaseUrl + "/recommend/similar";

        try {
            // 요청 바디 생성
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("recipe_id", recipeId);
            requestBody.put("top_k", topK);

            // 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

            // Flask API 호출
            ResponseEntity<RecommendationResponse> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    entity,
                    RecommendationResponse.class
            );

            log.info("유사 레시피 추천 성공 - recipeId: {}", recipeId);

            return response.getBody();

        } catch (Exception e) {
            log.error("유사 레시피 추천 실패 - recipeId: {}, error: {}", recipeId, e.getMessage());
            throw new RuntimeException("추천 시스템 오류", e);
        }
    }

    /**
     * 사용자 피드백 전송 (비동기)
     */
    public void sendFeedback(Long userId, Long recipeId, String interactionType) {
        String url = flaskBaseUrl + "/feedback";

        // 비동기 처리
        new Thread(() -> {
            try {
                Map<String, Object> requestBody = new HashMap<>();
                requestBody.put("user_id", userId);
                requestBody.put("recipe_id", recipeId);
                requestBody.put("interaction_type", interactionType);

                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);

                HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

                restTemplate.exchange(url, HttpMethod.POST, entity, Void.class);

                log.debug("피드백 전송 성공 - userId: {}, recipeId: {}, type: {}",
                        userId, recipeId, interactionType);

            } catch (Exception e) {
                log.warn("피드백 전송 실패 (무시): {}", e.getMessage());
            }
        }).start();
    }
}