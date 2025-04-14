package com.example.restfulpolygon.service;

import com.example.restfulpolygon.entity.StockTick;
import com.example.restfulpolygon.repository.StockTickRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.neovisionaries.ws.client.*;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.event.EventListener;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional; // Important for saving data


import java.io.IOException;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import com.example.restfulpolygon.entity.StockAggregate; //
import com.example.restfulpolygon.repository.StockAggregateRepository;


@Service
public class PolygonWebSocketService {

    private static final Logger log = LoggerFactory.getLogger(PolygonWebSocketService.class);
    private static final String POLYGON_WS_URL = "wss://socket.polygon.io/stocks";

    @Value("a5etjp2CyNlCcop5tw2GMyo40HKRnJYH") // Read API key from application.properties or environment
    private String apiKey;

    @Value("TSLA") // Read tickers from application.properties or environment
    private String tickersToSubscribe; // Expecting comma-separated like "TSLA,AAPL"

    @Autowired
    private StockTickRepository stockTickRepository;
    
    @Autowired // <-- ADD THIS
    private StockAggregateRepository stockAggregateRepository;

    private WebSocket ws;
    private final WebSocketFactory factory = new WebSocketFactory();
    private final ObjectMapper objectMapper = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false); // Ignore fields we don't map

    private final ScheduledExecutorService reconnectScheduler = Executors.newSingleThreadScheduledExecutor();
    private volatile boolean explicitlyClosed = false; // Flag to prevent reconnect on manual shutdown

    // Inject other repositories if you add Quote/Aggregate entities later
    // @Autowired
    // private StockQuoteRepository stockQuoteRepository;


    // Using @EventListener ensures Spring is fully initialized before we try to connect
    @EventListener(ApplicationReadyEvent.class)
    public void connectOnStartup() {
        if (apiKey == null || apiKey.isEmpty() || apiKey.equals("YOUR_NEW_SECURE_API_KEY") ) { // Basic check
             log.error("!!!!!! Polygon API Key is missing or placeholder. WebSocket connection NOT started. Set 'polygon.api.key' !!!!!");
             return;
        }
         if (tickersToSubscribe == null || tickersToSubscribe.isEmpty()) {
             log.error("!!!!!! No tickers configured to subscribe. WebSocket connection NOT started. Set 'polygon.tickers' !!!!!");
             return;
         }
        log.info("Application ready. Initiating WebSocket connection...");
        explicitlyClosed = false; // Ensure flag is reset on startup
        connect();
    }

    private void connect() {
        if (explicitlyClosed) {
            log.info("Explicitly closed, not attempting to connect.");
            return;
        }

        try {
            log.info("Attempting to connect to Polygon WebSocket: {}", POLYGON_WS_URL);
            if (ws != null && ws.isOpen()) {
                 log.warn("WebSocket is already open or connecting. Skipping new connection attempt.");
                 return;
            }

            ws = factory.createSocket(POLYGON_WS_URL);
            ws.addListener(new PolygonWebSocketListener()); // Add the inner listener class
            ws.setPingInterval(30 * 1000); // Keepalive ping
            ws.connectAsynchronously(); // Don't block startup thread


        } catch (IOException e) {
            log.error("Failed to create WebSocket socket", e);
            scheduleReconnect(); // Attempt to reconnect on creation failure
        }
    }

    @PreDestroy // Called when Spring shuts down the application
    public void disconnect() {
         explicitlyClosed = true; // Set flag *before* disconnecting
         log.info("Application shutting down. Disconnecting WebSocket...");
        if (ws != null) {
             ws.sendClose(1000, "Client shutdown");
             // Force disconnect if close doesn't happen quickly, cleanup handled by listener
             ws.disconnect(1000); // Timeout for disconnection
        }
         // Shutdown reconnect scheduler
        reconnectScheduler.shutdown();
        try {
            if (!reconnectScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                reconnectScheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            reconnectScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
        log.info("WebSocket disconnect initiated.");
    }

    private void scheduleReconnect() {
        if (explicitlyClosed) {
            log.info("Explicitly closed, skipping reconnect schedule.");
            return;
        }
        log.warn("Scheduling WebSocket reconnect attempt in 15 seconds...");
        try {
            reconnectScheduler.schedule(this::connect, 15, TimeUnit.SECONDS);
        } catch (Exception e) {
             log.error("Failed to schedule reconnect", e);
        }
    }


    // ----- Inner Listener Class -----
    private class PolygonWebSocketListener extends WebSocketAdapter {

        @Override
        public void onConnected(WebSocket websocket, Map<String, List<String>> headers) {
            log.info("WebSocket Connected! Authenticating...");
            sendAuthentication(websocket);
        }

        @Override
        public void onConnectError(WebSocket websocket, WebSocketException exception) {
             log.error("WebSocket Connect Error: {}", exception.getMessage(), exception);
             scheduleReconnect(); // Attempt to reconnect on connection error
        }

        @Override
        public void onDisconnected(WebSocket websocket, WebSocketFrame serverCloseFrame, WebSocketFrame clientCloseFrame, boolean closedByServer) {
             log.warn("WebSocket Disconnected. Server closing frame: {}, Client closing frame: {}, Closed by server: {}",
                    serverCloseFrame, clientCloseFrame, closedByServer);
            if (!explicitlyClosed) { // Only reconnect if not manually shut down
                scheduleReconnect();
            }
        }

        @Override
        public void onError(WebSocket websocket, WebSocketException cause) throws Exception {
            log.error("WebSocket Generic Error: {}", cause.getMessage(), cause);
             // Consider scheduling reconnect based on error type? Maybe not always desirable.
             // Let onDisconnected handle most reconnection triggers
        }

        @Override
        public void onTextMessage(WebSocket websocket, String text) {
            // log.debug("Raw message received: {}", text); // Too verbose for production normally
            try {
                List<Map<String, Object>> messages = objectMapper.readValue(text, new TypeReference<List<Map<String, Object>>>() {});
                for (Map<String, Object> message : messages) {
                    processMessage(message);
                }
            } catch (JsonProcessingException e) {
                log.error("Failed to parse JSON message: {}", text, e);
            } catch (Exception e) {
                log.error("Error processing message map: {}", e.getMessage(), e);
            }
        }
    }

    // --- Message Processing ---

    private void sendAuthentication(WebSocket websocket) {
        String authMsg = String.format("{\"action\":\"auth\",\"params\":\"%s\"}", apiKey);
        log.info("Sending authentication message...");
        websocket.sendText(authMsg);
    }

    private void sendSubscription(WebSocket websocket) {
        String[] tickers = tickersToSubscribe.split(",");
        if (tickers.length > 0) {
            // Subscribe to Trades (T) AND Aggregates Minute (AM)
            String tradeParams = List.of(tickers).stream().map(t -> "T." + t.trim().toUpperCase()).collect(Collectors.joining(","));
            String aggParams = List.of(tickers).stream().map(t -> "AM." + t.trim().toUpperCase()).collect(Collectors.joining(","));
            String allParams = tradeParams + "," + aggParams; // Combine params

            String subMsg = String.format("{\"action\":\"subscribe\",\"params\":\"%s\"}", allParams);
            log.info("Sending subscription message for: {}", allParams);
            websocket.sendText(subMsg);
        } else {
            log.warn("No valid tickers found in configuration '{}' to subscribe to.", tickersToSubscribe);
        }
     // TODO: Add subscriptions for Q. (Quotes) here if needed later
    }


    private void processMessage(Map<String, Object> message) {
        String eventType = (String) message.get("ev");
        if (eventType == null) return;

        switch (eventType) {
            case "status":
                handleStatusMessage(message);
                break;
            case "T": // Trade (Tick)
                handleTradeMessage(message);
                break;
            case "AM": // <-- ADD THIS CASE
                handleAggregateMessage(message);
                break;
            // TODO: Add cases for "Q" (Quotes)
            default:
                log.trace("Received unhandled event type: {}", eventType);
                break;
        }
    }

    private void handleStatusMessage(Map<String, Object> message) {
        String status = (String) message.get("status");
        String statusMessage = (String) message.get("message");
        log.info("Status received: {} - {}", status, statusMessage);

        if ("auth_success".equalsIgnoreCase(status)) {
            log.info("Authentication successful!");
             // Now send subscription message AFTER successful auth
            sendSubscription(ws); // Send subscription using the class member 'ws'
        } else if ("auth_failed".equalsIgnoreCase(status)) {
            log.error("!!!!!! Authentication FAILED: {} !!!!! Please check API key.", statusMessage);
            // Maybe disconnect explicitly on auth failure?
             disconnect();
        } else if ("subscribed".equalsIgnoreCase(status)) {
            log.info("Successfully subscribed to channels: {}", message.get("channels"));
        }
    }

    @Transactional // Ensure save operations are transactional
    public void handleTradeMessage(Map<String, Object> message) {
        try {
             // Basic mapping - Assumes field names match common Polygon 'T' event
            StockTick tick = new StockTick();
            tick.setSymbol((String) message.get("sym"));
            tick.setPrice(((Number) message.get("p")).doubleValue()); // Handle potential integer/double
            tick.setSize(((Number) message.get("s")).longValue());     // Handle potential integer/long
            // Polygon timestamps are often UNIX Nanoseconds or Milliseconds for trades. Check API doc!
            // Assuming Nanoseconds here based on v3 spec for example, adjust if using Milliseconds
             long timestampNanos = ((Number) message.get("t")).longValue();
             tick.setTimestamp(Instant.ofEpochSecond(0, timestampNanos));


            // Optional fields
            if (message.containsKey("x")) {
                tick.setExchange(((Number) message.get("x")).intValue());
            }
            if (message.containsKey("c")) {
                // Conditions usually an array of ints, store as comma-separated string or JSON string for now
                Object condObj = message.get("c");
                 if (condObj instanceof List) {
                     tick.setConditions(((List<?>)condObj).stream().map(Object::toString).collect(Collectors.joining(",")));
                 } else {
                     tick.setConditions(condObj != null ? condObj.toString() : null);
                 }
            }

            // SAVE TO DATABASE
            stockTickRepository.save(tick);
            log.trace("Saved tick: {}", tick); // Log tick saving if trace is enabled

        } catch (Exception e) {
            log.error("Error processing or saving trade message: {} - Data: {}", e.getMessage(), message, e);
             // Decide if you want to throw or just log - throwing might break processing loop
        }
    }
    
    @Transactional // Ensure save operations are transactional
    public void handleAggregateMessage(Map<String, Object> message) {
        try {
            StockAggregate agg = new StockAggregate();
            agg.setSymbol((String) message.get("sym"));
            agg.setOpen(((Number) message.get("o")).doubleValue());
            agg.setHigh(((Number) message.get("h")).doubleValue());
            agg.setLow(((Number) message.get("l")).doubleValue());
            agg.setClose(((Number) message.get("c")).doubleValue());
            agg.setVolume(((Number) message.get("v")).longValue());

            // Polygon timestamps are usually UNIX Milliseconds for aggregates
            long startMillis = ((Number) message.get("s")).longValue();
            agg.setStartTime(Instant.ofEpochMilli(startMillis));
            // Optional fields
            if (message.containsKey("vw")) { // Volume Weighted Average Price
                agg.setVwap(((Number) message.get("vw")).doubleValue());
            }
            if (message.containsKey("e")) { // End timestamp
                long endMillis = ((Number) message.get("e")).longValue();
                agg.setEndTime(Instant.ofEpochMilli(endMillis));
            }
            if (message.containsKey("n")) { // Number of trades
                agg.setNumTrades(((Number) message.get("n")).intValue());
            }


            // SAVE TO DATABASE
            stockAggregateRepository.save(agg);
            log.trace("Saved aggregate: {}", agg);

        } catch (Exception e) {
            log.error("Error processing or saving aggregate message: {} - Data: {}", e.getMessage(), message, e);
        }
    }

    // --- TODO: Implement handlers for Quotes (Q) or Aggregates (AM) ---
    /*
    @Transactional
    private void handleQuoteMessage(Map<String, Object> message) {
         log.debug("Processing Quote (Q): {}", message);
        // 1. Define StockQuote entity & repository
        // 2. Parse fields (bp, bs, ap, as, t, etc.)
        // 3. Create StockQuote object
        // 4. Save using stockQuoteRepository.save()
    }

    @Transactional
    private void handleAggregateMessage(Map<String, Object> message) {
         log.debug("Processing Aggregate Minute (AM): {}", message);
        // 1. Define StockAggregate entity & repository
        // 2. Parse fields (o, h, l, c, v, vw, s, e, etc.)
        // 3. Create StockAggregate object
        // 4. Save using stockAggregateRepository.save()
    }
    */
}
