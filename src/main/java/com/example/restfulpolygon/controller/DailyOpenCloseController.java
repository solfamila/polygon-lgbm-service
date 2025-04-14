package com.example.restfulpolygon.controller;

import com.example.restfulpolygon.entity.DailyOpenCloseEntity;
import com.example.restfulpolygon.service.DailyOpenCloseService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@RestController
@RequestMapping("/api")
public class DailyOpenCloseController {

    private final ObjectMapper objectMapper;
    private final RestTemplate restTemplate;

    @Autowired
    private DailyOpenCloseService service;

    @Value("${polygon.api.key}")
    private String apiKey;

    @Autowired
    public DailyOpenCloseController(ObjectMapper objectMapper, RestTemplate restTemplate) {
        this.objectMapper = objectMapper;
        this.restTemplate = restTemplate;
    }

    @GetMapping("/daily-open-close/{ticker}/{date}")
    public ResponseEntity<DailyOpenCloseEntity> getDailyOpenClose(@PathVariable String ticker, @PathVariable String date) {
        String url = String.format("https://api.polygon.io/v1/open-close/%s/%s?adjusted=true&apiKey=%s",
                ticker, date, apiKey);

        ResponseEntity<DailyOpenCloseEntity> response = restTemplate.getForEntity(url, DailyOpenCloseEntity.class);

        if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
            DailyOpenCloseEntity entity = response.getBody();
            entity.setFrom(date); // Set the date as String directly

            // Save the entity using the service layer
            DailyOpenCloseEntity savedEntity = service.save(entity);
            return ResponseEntity.ok(savedEntity);
        } else {
            // Handle error or throw an exception
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    @GetMapping
    public List<DailyOpenCloseEntity> getAllDailyOpenClose() {
        return service.findAll();
    }

    @GetMapping("/{id}")
    public ResponseEntity<DailyOpenCloseEntity> getDailyOpenCloseById(@PathVariable Long id) {
        return service.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public DailyOpenCloseEntity createDailyOpenClose(@RequestBody DailyOpenCloseEntity dailyOpenCloseEntity) {
        return service.save(dailyOpenCloseEntity);
    }

    @PutMapping("/{id}")
    public ResponseEntity<DailyOpenCloseEntity> updateDailyOpenClose(@PathVariable Long id, @RequestBody DailyOpenCloseEntity dailyOpenCloseEntity) {
        return service.findById(id)
                .map(existing -> {
                    dailyOpenCloseEntity.setId(existing.getId());
                    service.save(dailyOpenCloseEntity);
                    return ResponseEntity.ok(dailyOpenCloseEntity);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteDailyOpenClose(@PathVariable Long id) {
        return service.findById(id)
                .map(entity -> {
                    service.delete(entity.getId());
                    return ResponseEntity.ok().build();
                })
                .orElse(ResponseEntity.notFound().build());
    }

}
