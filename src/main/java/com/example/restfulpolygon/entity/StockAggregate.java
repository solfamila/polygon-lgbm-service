package com.example.restfulpolygon.entity;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "stock_aggregates_min") // Table name for minute aggregates
public class StockAggregate {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id; // Generic ID

    @Column(nullable = false)
    private String symbol; // Ticker symbol

    @Column(nullable = false, name = "agg_open") // Use distinct names to avoid keyword clash if any
    private double open;

    @Column(nullable = false, name = "agg_high")
    private double high;

    @Column(nullable = false, name = "agg_low")
    private double low;

    @Column(nullable = false, name = "agg_close")
    private double close;

    @Column(nullable = false)
    private long volume;

    @Column
    private Double vwap; // VWAP might be null sometimes

    @Column(nullable = false, columnDefinition = "TIMESTAMPTZ")
    private Instant startTime; // Start timestamp of the aggregate window

    @Column(columnDefinition = "TIMESTAMPTZ") // Allow null just in case
    private Instant endTime;   // End timestamp (optional to store)

    @Column // Number of trades in window (from 'n' field)
    private Integer numTrades;

    // --- Constructors ---
    public StockAggregate() {}

    // --- Getters and Setters --- (Generate or write as needed)


    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public double getOpen() {
        return open;
    }

    public void setOpen(double open) {
        this.open = open;
    }

    public double getHigh() {
        return high;
    }

    public void setHigh(double high) {
        this.high = high;
    }

    public double getLow() {
        return low;
    }

    public void setLow(double low) {
        this.low = low;
    }

    public double getClose() {
        return close;
    }

    public void setClose(double close) {
        this.close = close;
    }

    public long getVolume() {
        return volume;
    }

    public void setVolume(long volume) {
        this.volume = volume;
    }

    public Double getVwap() {
        return vwap;
    }

    public void setVwap(Double vwap) {
        this.vwap = vwap;
    }

    public Instant getStartTime() {
        return startTime;
    }

    public void setStartTime(Instant startTime) {
        this.startTime = startTime;
    }

    public Instant getEndTime() {
        return endTime;
    }

    public void setEndTime(Instant endTime) {
        this.endTime = endTime;
    }

    public Integer getNumTrades() {
        return numTrades;
    }

    public void setNumTrades(Integer numTrades) {
        this.numTrades = numTrades;
    }

     @Override
    public String toString() {
        return "StockAggregate{" +
                "symbol='" + symbol + '\'' +
                ", startTime=" + startTime +
                ", open=" + open +
                ", high=" + high +
                ", low=" + low +
                ", close=" + close +
                ", volume=" + volume +
                '}';
    }
}
