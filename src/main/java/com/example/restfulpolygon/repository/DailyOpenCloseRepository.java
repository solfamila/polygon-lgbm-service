package com.example.restfulpolygon.repository;

import com.example.restfulpolygon.entity.DailyOpenCloseEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DailyOpenCloseRepository extends JpaRepository<DailyOpenCloseEntity, Long> {
    // Repository queries if needed
}
