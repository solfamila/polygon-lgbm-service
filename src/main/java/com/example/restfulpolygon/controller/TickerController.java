package com.example.restfulpolygon.controller;

import com.example.restfulpolygon.entity.TickerDetailsEntity;
import com.example.restfulpolygon.service.TickerDetailsService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class TickerController {

    private final TickerDetailsService tickerDetailsService;

    public TickerController(TickerDetailsService tickerDetailsService) {
        this.tickerDetailsService = tickerDetailsService;
    }

    @PostMapping("/search")
    public String searchTicker(@RequestParam String ticker, Model model) {
        TickerDetailsEntity tickerDetails = tickerDetailsService.getTickerDetails(ticker);
        model.addAttribute("tickerDetails", tickerDetails);
        return "index";  // Assuming you want to show the results on the same page
    }
}

