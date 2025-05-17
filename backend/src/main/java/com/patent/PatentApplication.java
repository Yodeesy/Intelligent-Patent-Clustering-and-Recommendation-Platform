package com.patent;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;

@SpringBootApplication
@OpenAPIDefinition(
    info = @Info(
        title = "专利聚类与推荐平台 API",
        version = "1.0",
        description = "智能专利聚类与推荐平台的 REST API 文档"
    )
)
public class PatentApplication {
    public static void main(String[] args) {
        SpringApplication.run(PatentApplication.class, args);
    }
} 