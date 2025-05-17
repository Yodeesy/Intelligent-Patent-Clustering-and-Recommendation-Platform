package com.patent.controller;

import com.patent.model.Patent;
import com.patent.service.PatentService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/patents")
@Tag(name = "专利接口", description = "专利相关的操作接口")
@CrossOrigin(origins = "*")
public class PatentController {
    
    private final PatentService patentService;
    
    @Autowired
    public PatentController(PatentService patentService) {
        this.patentService = patentService;
    }
    
    @GetMapping("/random")
    @Operation(summary = "获取随机专利列表", description = "返回指定数量的随机专利")
    public ResponseEntity<List<Patent>> getRandomPatents(
            @RequestParam(defaultValue = "100") int size) {
        return ResponseEntity.ok(patentService.getRandomPatents(size));
    }
    
    @GetMapping("/{patentId}/similar")
    @Operation(summary = "获取相似专利", description = "根据专利ID获取相似的专利列表")
    public ResponseEntity<List<Patent>> getSimilarPatents(
            @PathVariable String patentId,
            @RequestParam(defaultValue = "10") int limit) {
        return ResponseEntity.ok(patentService.getSimilarPatents(patentId, limit));
    }
    
    @PostMapping("/cluster")
    @Operation(summary = "专利聚类", description = "对指定的专利列表进行聚类分析")
    public ResponseEntity<Map<String, Object>> clusterPatents(
            @RequestBody List<String> patentIds) {
        return ResponseEntity.ok(patentService.clusterPatents(patentIds));
    }
} 