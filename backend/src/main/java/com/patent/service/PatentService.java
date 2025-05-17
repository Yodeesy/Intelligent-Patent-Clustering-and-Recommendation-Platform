package com.patent.service;

import com.patent.model.Patent;
import com.patent.repository.PatentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Service
public class PatentService {
    
    private final PatentRepository patentRepository;
    private final RestTemplate restTemplate;
    
    @Value("${python.service.url}")
    private String pythonServiceUrl;
    
    @Autowired
    public PatentService(PatentRepository patentRepository, RestTemplate restTemplate) {
        this.patentRepository = patentRepository;
        this.restTemplate = restTemplate;
    }
    
    public List<Patent> getRandomPatents(int size) {
        return patentRepository.findRandomPatents(size);
    }
    
    public List<Patent> getSimilarPatents(String patentId, int limit) {
        return patentRepository.findSimilarPatents(patentId, limit);
    }
    
    public Map<String, Object> clusterPatents(List<String> patentIds) {
        // 调用Python服务进行聚类
        return restTemplate.postForObject(
            pythonServiceUrl + "/cluster",
            patentIds,
            Map.class
        );
    }
} 