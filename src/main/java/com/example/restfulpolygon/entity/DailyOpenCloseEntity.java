package com.example.restfulpolygon.entity;

import lombok.Getter;
import lombok.Setter;
import jakarta.persistence.*;
import jakarta.validation.constraints.*;

@Entity
@Getter
@Setter
@Table(name = "daily_open_close")
public class DailyOpenCloseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "Symbol is mandatory")
    private String symbol;

    @NotNull(message = "Date is mandatory")
    @Column(name = "`date`")
    private String from;

    @NotNull(message = "Open price cannot be null")
    @DecimalMin(value = "0.0", inclusive = false, message = "Open price must be greater than 0")
    private double open;

    @NotNull(message = "Close price cannot be null")
    @DecimalMin(value = "0.0", inclusive = false, message = "Close price must be greater than 0")
    private double close;

    @NotNull(message = "High price cannot be null")
    @DecimalMin(value = "0.0", inclusive = false, message = "High price must be greater than 0")
    private double high;

    @NotNull(message = "Low price cannot be null")
    @DecimalMin(value = "0.0", inclusive = false, message = "Low price must be greater than 0")
    private double low;

    @NotNull(message = "Volume cannot be null")
    @Min(value = 1, message = "Volume must be greater than 0")
    private double volume;

    private double afterHours;
    private double preMarket;

    @NotBlank(message = "Status is mandatory")
    private String status;

    // Constructors, getters, and setters...
}
