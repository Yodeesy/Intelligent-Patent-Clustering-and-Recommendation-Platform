package com.patent.service;

import com.patent.model.Patent;
import com.patent.repository.PatentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
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
        return restTemplate.postForObject(
            pythonServiceUrl + "/cluster",
            patentIds,
            Map.class
        );
    }

    // ✅ 新增：基于新专利结构进行推荐
    public Map<String, Object> recommendByNewPatent(Map<String, Object> newPatent) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(newPatent, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(
            pythonServiceUrl + "/recommend",
            request,
            Map.class
        );

        return response.getBody();
    }
}
