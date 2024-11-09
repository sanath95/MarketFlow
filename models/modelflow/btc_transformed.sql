SELECT
    timestamp,
    open,
    close,
    high,
    low,
    volume,
    LAG(high) OVER (ORDER BY timestamp) AS lag,
    AVG(high) OVER (ORDER BY timestamp ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS moving_avg
FROM {{ ref('btc') }}