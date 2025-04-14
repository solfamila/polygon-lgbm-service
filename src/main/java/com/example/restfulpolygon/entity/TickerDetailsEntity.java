package com.example.restfulpolygon.entity;

import lombok.Getter;
import lombok.Setter;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;

import java.math.BigDecimal;

@Entity
@Getter
@Setter
@Table(name = "ticker_details")
public class TickerDetailsEntity {

    @Id
    private String ticker;
    private String name;
    private String market;
    private String locale;

    @Column(name = "primary_exchange")
    private String primaryExchange;
    private String type;
    private boolean active;

    @Column(name = "currency_name")
    private String currencyName;
    private String cik;

    @Column(name = "composite_figi")
    private String compositeFigi;

    @Column(name = "share_class_figi")
    private String shareClassFigi;

    @Column(name = "market_cap")
    private BigDecimal marketCap;

    @Column(name = "phone_number")
    private String phoneNumber;

    private String address1;
    private String city;
    private String state;

    @Column(name = "postal_code")
    private String postalCode;

    private String description;
    private String sicCode;

    @Column(name = "sic_description")
    private String sicDescription;

    @Column(name = "ticker_root")
    private String tickerRoot;

    @Column(name = "homepage_url")
    private String homepageUrl;

    @Column(name = "total_employees")
    private Integer totalEmployees;

    @Column(name = "list_date")
    private String listDate;

    @Column(name = "logo_url")
    private String logoUrl;

    @Column(name = "icon_url")
    private String iconUrl;

    @Column(name = "share_class_shares_outstanding")
    private Long shareClassSharesOutstanding;

    @Column(name = "weighted_shares_outstanding")
    private Long weightedSharesOutstanding;

    @Column(name = "round_lot")
    private Integer roundLot;


    // Assume other fields are included with appropriate types and annotations...

    // Constructors, getters, and setters...
}
