package com.patent.model;

import lombok.Data;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

@Data
@Node("User")
public class User {
    @Id
    @GeneratedValue
    private Long id;
    
    private String username;
    private String email;
    private String password;
    private List<String> roles;
    
    // 默认构造函数
    public User() {
        this.roles = new ArrayList<>();
        this.roles.add("ROLE_USER");
    }
    
    // 带参数的构造函数
    public User(String username, String email, String password) {
        this();
        this.username = username;
        this.email = email;
        this.password = password;
    }

    // 显式定义方法以确保与 UserPrincipal 期望的方法名完全匹配
    public Long getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public String getPassword() {
        return password;
    }

    public Collection<String> getRoles() {
        return roles;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public void setRoles(List<String> roles) {
        this.roles = roles;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
} 