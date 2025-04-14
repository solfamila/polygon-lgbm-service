package com.example.restfulpolygon.service;

import com.example.restfulpolygon.entity.TickerDetailsEntity;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.math.BigDecimal;

@Service
public class TickerDetailsService {

    private final RestTemplate restTemplate;

    @Value("${polygon.api.key}")
    private String apiKey;

    public TickerDetailsService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public TickerDetailsEntity getTickerDetails(String ticker) {
        String url = "https://api.polygon.io/v3/reference/tickers/" + ticker + "?apiKey=" + apiKey;
        ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);

        // Parse the response into your entity
        // You might use a library like Jackson or Gson to convert JSON to your entity
        TickerDetailsEntity tickerDetailsEntity = parseResponseToEntity(response.getBody());

        return tickerDetailsEntity;
    }

    private TickerDetailsEntity parseResponseToEntity(String jsonResponse) {
        Gson gson = new GsonBuilder()
                .registerTypeAdapter(BigDecimal.class, new BigDecimalDeserializer())
                .create();
        TickerDetailsResponse response = gson.fromJson(jsonResponse, TickerDetailsResponse.class);
        return mapResponseToEntity(response.getResults());
    }


    private TickerDetailsEntity mapResponseToEntity(TickerDetailsResponse.Results results) {
        if (results == null) {
            return null; // or throw an exception if results should never be null
        }

        TickerDetailsEntity entity = new TickerDetailsEntity();

        // Mapping the properties from Results to TickerDetailsEntity
        entity.setTicker(results.getTicker());
        entity.setName(results.getName());
        entity.setMarket(results.getMarket());
        entity.setLocale(results.getLocale());
        entity.setPrimaryExchange(results.getPrimaryExchange());
        entity.setActive(results.isActive());
        entity.setCurrencyName(results.getCurrencyName());
        entity.setCik(results.getCik());
        entity.setMarketCap(results.getMarketCap());
        entity.setPhoneNumber(results.getPhoneNumber());

        // Mapping the address, which is a nested object
        if (results.getAddress() != null) {
            entity.setAddress1(results.getAddress().getAddress1());
            entity.setCity(results.getAddress().getCity());
            entity.setState(results.getAddress().getState());
            entity.setPostalCode(results.getAddress().getPostalCode());
        }

        entity.setDescription(results.getDescription());
        entity.setSicCode(results.getSicCode());
        entity.setSicDescription(results.getSicDescription());
        entity.setTickerRoot(results.getTickerRoot());
        entity.setHomepageUrl(results.getHomepageUrl());
        entity.setTotalEmployees(results.getTotalEmployees());
        entity.setListDate(results.getListDate());

        // Mapping the branding, another nested object
        if (results.getBranding() != null) {
            entity.setLogoUrl(results.getBranding().getLogoUrl());
            entity.setIconUrl(results.getBranding().getIconUrl());
        }

        entity.setShareClassSharesOutstanding(results.getShareClassSharesOutstanding());
        entity.setWeightedSharesOutstanding(results.getWeightedSharesOutstanding());
        entity.setRoundLot(results.getRoundLot());

        return entity;
    }

}
