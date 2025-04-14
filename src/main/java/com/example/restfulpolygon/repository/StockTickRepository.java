package com.example.restfulpolygon.repository;

import com.example.restfulpolygon.entity.StockTick;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StockTickRepository extends JpaRepository<StockTick, Long> {
    // Spring Data JPA automatically provides save(), findById(), findAll(), etc.
    // We can add custom query methods later if needed.
}
