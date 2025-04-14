plugins {
    java
    id("org.springframework.boot") version "3.1.5"
    id("io.spring.dependency-management") version "1.1.3"
}

group = "com.example"
version = "0.0.1-SNAPSHOT"

java {
    sourceCompatibility = JavaVersion.VERSION_21
}

configurations {
    compileOnly {
        extendsFrom(configurations.annotationProcessor.get())
    }
}

repositories {
    mavenCentral()
    maven(url = "https://jitpack.io")
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-websocket")
    compileOnly("org.projectlombok:lombok")
    // NOTE: This project uses TimescaleDB/PostgreSQL via docker-compose, not MySQL.
    // You might want to comment out or remove mysql-connector-j unless needed elsewhere.
    // runtimeOnly("com.mysql:mysql-connector-j")
    // And potentially add the PostgreSQL driver:
    runtimeOnly("org.postgresql:postgresql") 
    annotationProcessor("org.projectlombok:lombok")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    implementation ("com.github.polygon-io:client-jvm:v5.1.0")
    implementation("jakarta.persistence:jakarta.persistence-api:3.1.0")
    implementation ("com.fasterxml.jackson.core:jackson-databind")
    implementation ("jakarta.validation:jakarta.validation-api:3.0.0")
    implementation ("org.springframework.boot:spring-boot-starter-thymeleaf") // Note: Might not be strictly needed if no UI is served by this backend
    implementation ("com.google.code.gson:gson:2.10.1")
    implementation("com.neovisionaries:nv-websocket-client:2.14") // Or check for the latest version
}

tasks.withType<Test> {
    useJUnitPlatform()
}

// Explicitly configure buildpacks builder and run image (ADDED BLOCK)
tasks.withType<org.springframework.boot.gradle.tasks.bundling.BootBuildImage> {
    builder.set("paketobuildpacks/builder-jammy-base:latest") // Use Jammy base builder
    runImage.set("paketobuildpacks/run-jammy-base:latest")    // Use Jammy base run image
    // The imageName derived from settings.gradle.kts ('polygon-ticker-service') should still be active
}
