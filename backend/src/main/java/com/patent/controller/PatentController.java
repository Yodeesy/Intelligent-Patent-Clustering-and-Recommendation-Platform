package com.patent.controller;

import com.patent.model.Patent;
import com.patent.service.PatentService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/patents")
@Api(tags = "专利接口")
@CrossOrigin(origins = "*")
public class PatentController {
    
    private final PatentService patentService;
    
    @Autowired
    public PatentController(PatentService patentService) {
        this.patentService = patentService;
    }
    
    @GetMapping("/random")
    @ApiOperation("获取随机专利列表")
    public ResponseEntity<List<Patent>> getRandomPatents(
            @RequestParam(defaultValue = "100") int size) {
        return ResponseEntity.ok(patentService.getRandomPatents(size));
    }
    
    @GetMapping("/{patentId}/similar")
    @ApiOperation("获取相似专利")
    public ResponseEntity<List<Patent>> getSimilarPatents(
            @PathVariable String patentId,
            @RequestParam(defaultValue = "10") int limit) {
        return ResponseEntity.ok(patentService.getSimilarPatents(patentId, limit));
    }
    
    @PostMapping("/cluster")
    @ApiOperation("专利聚类")
    public ResponseEntity<Map<String, Object>> clusterPatents(
            @RequestBody List<String> patentIds) {
        return ResponseEntity.ok(patentService.clusterPatents(patentIds));
    }
} 