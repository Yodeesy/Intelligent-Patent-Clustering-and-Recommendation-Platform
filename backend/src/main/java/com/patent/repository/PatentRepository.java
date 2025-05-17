package com.patent.repository;

import com.patent.model.Patent;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatentRepository extends Neo4jRepository<Patent, String> {
    
    @Query("MATCH (p:Patent) RETURN p LIMIT $size")
    List<Patent> findRandomPatents(int size);
    
    @Query("MATCH (p1:Patent {id: $patentId})-[r:SIMILAR_TO]-(p2:Patent) " +
           "WHERE id(p1) < id(p2) OR (id(p1) = id(p2) AND type(r) = 'SIMILAR_TO') " +
           "RETURN p2 ORDER BY r.similarity DESC LIMIT $limit")
    List<Patent> findSimilarPatents(String patentId, int limit);
} 