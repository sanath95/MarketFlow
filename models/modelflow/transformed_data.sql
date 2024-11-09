WITH btc_data AS (
    SELECT
        timestamp,
        high,
        LAG(high) OVER (ORDER BY timestamp) AS lag,
        AVG(high) OVER (ORDER BY timestamp ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS moving_avg,
        "BTC-USD" as trading_pair
    FROM  {{ ref('btc') }}
),
xau_data AS (
    SELECT
        timestamp,
        high,
        LAG(high) OVER (ORDER BY timestamp) AS lag,
        AVG(high) OVER (ORDER BY timestamp ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS moving_avg,
        "XAU-USD" as trading_pair
    FROM  {{ ref('xau') }}
)

SELECT * FROM btc_data
UNION ALL
SELECT * FROM xau_data
ORDER BY timestamp