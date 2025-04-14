package com.example.restfulpolygon.service;

import com.example.restfulpolygon.entity.DailyOpenCloseEntity;
import com.example.restfulpolygon.repository.DailyOpenCloseRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class DailyOpenCloseService {

    private final RestTemplate restTemplate;

    @Autowired
    private DailyOpenCloseRepository repository;

    @Value("${polygon.api.key}")
    private String apiKey;

    public DailyOpenCloseService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public DailyOpenCloseEntity getDailyOpenClose(String ticker, String date) {
        String url = String.format("https://api.polygon.io/v1/open-close/%s/%s?adjusted=true&apiKey=%s",
                ticker, date, apiKey);

        ResponseEntity<DailyOpenCloseEntity> response = restTemplate.getForEntity(url, DailyOpenCloseEntity.class);
        return response.getBody();
    }

    public List<DailyOpenCloseEntity> findAll() {
        return repository.findAll();
    }

    public Optional<DailyOpenCloseEntity> findById(Long id) {
        return repository.findById(id);
    }

    public DailyOpenCloseEntity save(DailyOpenCloseEntity dailyOpenCloseEntity) {
        return repository.save(dailyOpenCloseEntity);
    }

    public void update(DailyOpenCloseEntity dailyOpenCloseEntity) {
        repository.save(dailyOpenCloseEntity);
    }

    public void delete(Long id) {
        repository.deleteById(id);
    }
}



