package com.example.restfulpolygon.service;

import com.google.gson.annotations.SerializedName;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;

@Data
@NoArgsConstructor
public class TickerDetailsResponse {
    @SerializedName("request_id")
    private String requestId;

    private Results results;
    private String status;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Results {
        private String ticker;
        private String name;
        private String market;
        private String locale;
        @SerializedName("primary_exchange")
        private String primaryExchange;
        private boolean active;
        @SerializedName("currency_name")
        private String currencyName;
        private String cik;
        @SerializedName("market_cap")
        private BigDecimal marketCap;
        @SerializedName("phone_number")
        private String phoneNumber;
        private Address address;
        private String description;
        @SerializedName("sic_code")
        private String sicCode;
        @SerializedName("sic_description")
        private String sicDescription;
        @SerializedName("ticker_root")
        private String tickerRoot;
        @SerializedName("homepage_url")
        private String homepageUrl;
        @SerializedName("total_employees")
        private int totalEmployees;
        @SerializedName("list_date")
        private String listDate;
        private Branding branding;
        @SerializedName("share_class_shares_outstanding")
        private long shareClassSharesOutstanding;
        @SerializedName("weighted_shares_outstanding")
        private long weightedSharesOutstanding;
        @SerializedName("round_lot")
        private int roundLot;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Address {
        private String address1;
        private String city;
        private String state;
        @SerializedName("postal_code")
        private String postalCode;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Branding {
        @SerializedName("logo_url")
        private String logoUrl;
        @SerializedName("icon_url")
        private String iconUrl;
    }
}
