package com.example.restfulpolygon.controller;
import com.example.restfulpolygon.entity.TickerDetailsEntity;
import com.example.restfulpolygon.service.TickerDetailsService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TickerRestController {

    private final TickerDetailsService tickerDetailsService;

    public TickerRestController(TickerDetailsService tickerDetailsService) {
        this.tickerDetailsService = tickerDetailsService;
    }

    // Add this method to your existing controller class
    @GetMapping("/api/ticker-details/{ticker}")
    public TickerDetailsEntity getTickerDetails(@PathVariable String ticker) {
        return tickerDetailsService.getTickerDetails(ticker);
    }
}

