package com.example.restfulpolygon.service;

import com.google.gson.*;
import java.lang.reflect.Type;
import java.math.BigDecimal;

public class BigDecimalDeserializer implements JsonDeserializer<BigDecimal> {
    @Override
    public BigDecimal deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) throws JsonParseException {
        try {
            return json.getAsBigDecimal();
        } catch (NumberFormatException e) {
            // You could log this exception or handle it as needed for your use case
            throw new JsonParseException("Could not parse to BigDecimal: " + json, e);
        }
    }
}

