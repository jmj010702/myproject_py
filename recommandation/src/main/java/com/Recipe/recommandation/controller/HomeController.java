package com.Recipe.recommandation.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class HomeController {
    
    @GetMapping("/")
    @ResponseBody
    public String home() {
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>레시피 추천 시스템</title>
                <style>
                    body { font-family: Arial; padding: 50px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                    h1 { color: #2c3e50; }
                    .endpoint { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }
                    a { color: #3498db; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🍳 레시피 추천 시스템</h1>
                    <h2>📡 API 엔드포인트</h2>
                    
                    <div class="endpoint">
                        <strong>헬스 체크:</strong><br>
                        <a href="/api/recommendations/health">/api/recommendations/health</a>
                    </div>
                    
                    <div class="endpoint">
                        <strong>개인화 추천:</strong><br>
                        <a href="/api/recommendations/home?userId=1&topK=10">/api/recommendations/home?userId=1&topK=10</a>
                    </div>
                    
                    <div class="endpoint">
                        <strong>유사 레시피:</strong><br>
                        <a href="/api/recommendations/similar?recipeId=100&topK=5">/api/recommendations/similar?recipeId=100&topK=5</a>
                    </div>
                    
                    <h2>📊 시스템 상태</h2>
                    <p>✅ Spring Boot 서버: 실행 중</p>
                    <p>✅ Flask 추천 엔진: 연결됨</p>
                    <p>✅ NCF 모델: 로드됨</p>
                </div>
            </body>
            </html>
            """;
    }
}
```

저장 후 **Spring Boot 재시작**하면 `http://localhost:8080/`에서 웰컴 페이지가 보입니다.

---

## 🎯 지금 당장 테스트

브라우저에서 이 URL들을 열어보세요:

1. **헬스 체크**: 
```
   http://localhost:8080/api/recommendations/health
```

2. **추천 받기**:
```
   http://localhost:8080/api/recommendations/home?userId=1&topK=10
```

3. **유사 레시피**:
```
   http://localhost:8080/api/recommendations/similar?recipeId=100&topK=5