document.addEventListener('DOMContentLoaded', function () {
    const searchForm = document.getElementById('searchForm');

    searchForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const ticker = searchForm.search.value.trim();
        if (ticker) {
            const apiUrl = `http://localhost:8080/api/ticker-details/${ticker}`;

            fetch(apiUrl)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => displayResults(data))
                .catch(error => {
                    console.log(error);
                    document.getElementById('results').innerHTML = `<p>Error loading data!</p>`;
                });
        }
    });
    function formatMarketCap(value) {
        if (value < 1e3) return value.toFixed(2);
        if (value >= 1e3 && value < 1e6) return (value / 1e3).toFixed(2) + ' Thousand';
        if (value >= 1e6 && value < 1e9) return (value / 1e6).toFixed(2) + ' Million';
        if (value >= 1e9 && value < 1e12) return (value / 1e9).toFixed(2) + ' Billion';
        if (value >= 1e12) return (value / 1e12).toFixed(2) + ' Trillion';
    }

    function displayResults(data) {
        const resultsContainer = document.getElementById('results');

        // Debugging: Log the type and value of marketCap from the API
        console.log('Original marketCap from API:', data.marketCap, 'Type:', typeof data.marketCap);

        // Convert marketCap to a number if it's a string
        let marketCapValue = typeof data.marketCap === 'string' ? parseFloat(data.marketCap) : data.marketCap;

        // Debugging: Log the converted value and its type
        console.log('Converted marketCap:', marketCapValue, 'Type:', typeof marketCapValue);

        let formattedMarketCap = formatMarketCap(marketCapValue);

        // Additional Debugging: Log the formatted marketCap
        console.log('Formatted marketCap:', formattedMarketCap);

        resultsContainer.innerHTML = `
            <h3>Results for: ${data.ticker}</h3>
            <p>Name: ${data.name}</p>
            <p>Market: ${data.market}</p>
            <p>Locale: ${data.locale}</p>
            <p>Primary Exchange: ${data.primaryExchange}</p>
            <p>Active: ${data.active}</p>
            <p>Currency Name: ${data.currencyName}</p>
            <p>CIK: ${data.cik}</p>
            <p>Market Cap: ${formattedMarketCap}</p>
            <p>Phone Number: ${data.phoneNumber}</p>
            <p>Address: ${data.address}</p>
            <p>Description: ${data.description}</p>
            <!-- ... Additional fields as needed -->
        `;
    }
    // Temporary test
    console.log(formatMarketCap(1483553430926.16)); // Should output a formatted string
});
