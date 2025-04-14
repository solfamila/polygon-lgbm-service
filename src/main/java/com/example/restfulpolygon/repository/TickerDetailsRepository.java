package com.example.restfulpolygon.repository;

import com.example.restfulpolygon.entity.TickerDetailsEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TickerDetailsRepository extends JpaRepository<TickerDetailsEntity, String> {
    // Here, you can define methods to retrieve entities from the database.
    // Spring Data JPA will implement the method automatically.

    // For example, to find a TickerDetailsEntity by its ticker:
    //TickerDetailsEntity findByTicker(String ticker);
}

