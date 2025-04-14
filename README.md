# Polygon Ticker Service

## Description
The Polygon Ticker Service is a Java-based RESTful API service designed to provide detailed financial information about stock tickers. Utilizing the Polygon.io API, it offers insights into various aspects of stocks including company details, market capitalization, and other financial metrics crucial for investors and financial analysts.

## Features
- Real-time access to detailed ticker information.
- Integration with Polygon.io's financial database.
- Customizable queries for specific ticker data.
- Efficient JSON response parsing into Java entities.
- Caching for improved performance and reduced API calls.

## Prerequisites
Before utilizing this service, ensure you have:
- Java JDK 11 or higher installed.
- An active Polygon.io API key.
- A basic understanding of RESTful services and financial market terminologies.

## Setup and Installation
Clone the repository and build the project using Gradle:
```shell
git clone https://github.com/RomanPilyushin/Polygon-Ticker-Service.git
cd polygon-ticker-information-service
./gradlew build
```

## Usage
Start the service and make API calls to retrieve stock ticker information:
```shell
GET /ticker/AMZN
```
Fetches information for Amazon stock from Polygon.io.

## Configuration
Configure your API key in the application.properties:
```shell
polygon.api.key=YOUR_API_KEY_HERE
```

## Email
rpilyushin [at] gmail.com


