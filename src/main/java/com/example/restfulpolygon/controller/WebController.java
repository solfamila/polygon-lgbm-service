package com.example.restfulpolygon.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class WebController {

    @GetMapping("/")
    public String index() {
        return "index";  // Thymeleaf will map this to src/main/resources/templates/index.html
    }
}


