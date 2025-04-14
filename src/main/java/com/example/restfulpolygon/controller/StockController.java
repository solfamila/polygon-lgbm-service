package com.example.restfulpolygon.controller;

import com.example.restfulpolygon.entity.DailyOpenCloseEntity;
import com.example.restfulpolygon.service.DailyOpenCloseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/stocks")
public class StockController {

    private final DailyOpenCloseService service;

    @Autowired
    public StockController(DailyOpenCloseService service) {
        this.service = service;
    }

    @GetMapping("/open-close/{ticker}/{date}")
    public DailyOpenCloseEntity getDailyOpenClose(@PathVariable String ticker, @PathVariable String date) {
        return service.getDailyOpenClose(ticker, date);
    }
}

