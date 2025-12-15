// ============================================================================
// 📄 파일: RecommandationApplication.java (수정본)
// 위치: src/main/java/com/Recipe/recommandation/RecommandationApplication.java
// ============================================================================

package com.Recipe.recommandation;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RecommandationApplication {

	public static void main(String[] args) {
		SpringApplication.run(RecommandationApplication.class, args);

		System.out.println("\n" + "=".repeat(70));
		System.out.println("🍳 레시피 추천 시스템 - Spring Boot 서버 시작");
		System.out.println("=".repeat(70));
		System.out.println("📡 서버: http://localhost:8080");
		System.out.println("📊 API 엔드포인트:");
		System.out.println("  - GET  /api/recommendations/home?userId=1&topK=10");
		System.out.println("  - GET  /api/recommendations/similar?recipeId=456&topK=5");
		System.out.println("  - POST /api/recommendations/interaction");
		System.out.println("  - GET  /api/recommendations/health");
		System.out.println("=".repeat(70) + "\n");
	}
}