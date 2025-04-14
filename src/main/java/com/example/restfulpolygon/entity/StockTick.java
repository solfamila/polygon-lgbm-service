package com.example.restfulpolygon.entity;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "stock_ticks") // Define the table name explicitly
public class StockTick {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY) // Use DB auto-increment (PostgreSQL will handle this with bigserial)
    private Long id;

    @Column(nullable = false)
    private String symbol; // Ticker symbol (e.g., "TSLA")

    @Column(nullable = false)
    private double price; // Trade price

    @Column(nullable = false)
    private long size; // Trade size

    @Column(nullable = false, columnDefinition = "TIMESTAMPTZ") // Use TIMESTAMPTZ for timezone handling
    private Instant timestamp; // Trade timestamp (nanosecond precision)

    @Column // Example: Store the raw exchange ID if needed
    private Integer exchange;

    @Column // Store condition flags (can be complex, maybe JSON string or separate table later)
    private String conditions; // Keeping simple for now


    // --- Constructors ---
    public StockTick() {
    }

    // --- Getters and Setters --- (You can generate these with an IDE or Lombok if preferred, but showing explicit)

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

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public long getSize() {
        return size;
    }

    public void setSize(long size) {
        this.size = size;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Instant timestamp) {
        this.timestamp = timestamp;
    }

    public Integer getExchange() {
        return exchange;
    }

    public void setExchange(Integer exchange) {
        this.exchange = exchange;
    }

    public String getConditions() {
        return conditions;
    }

    public void setConditions(String conditions) {
        this.conditions = conditions;
    }

    // --- toString() --- useful for debugging
    @Override
    public String toString() {
        return "StockTick{" +
                "id=" + id +
                ", symbol='" + symbol + '\'' +
                ", price=" + price +
                ", size=" + size +
                ", timestamp=" + timestamp +
                '}';
    }
}
