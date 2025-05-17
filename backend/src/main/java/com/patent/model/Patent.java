package com.patent.model;

import lombok.Data;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Property;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.util.ArrayList;
import java.util.List;

@Data
@Node("Patent")
public class Patent {
    @Id
    private String id;
    
    @Property("title")
    private String title;
    
    @Property("abstract")
    private String abstractText;
    
    @Property("inventors")
    private List<String> inventors;
    
    @Property("applicants")
    private List<String> applicants;
    
    @Property("publicationDate")
    private String publicationDate;
    
    @Relationship(type = "SIMILAR_TO", direction = Relationship.Direction.OUTGOING)
    private List<Patent> similarPatents = new ArrayList<>();
    
    // 默认构造函数
    public Patent() {
        this.inventors = new ArrayList<>();
        this.applicants = new ArrayList<>();
        this.similarPatents = new ArrayList<>();
    }
    
    // 带参数的构造函数
    public Patent(String id, String title, String abstractText) {
        this();
        this.id = id;
        this.title = title;
        this.abstractText = abstractText;
    }
} 