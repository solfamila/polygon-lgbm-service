package com.example.restfulpolygon.repository;

import com.example.restfulpolygon.entity.StockAggregate;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StockAggregateRepository extends JpaRepository<StockAggregate, Long> {
    // Basic CRUD methods are provided
}
